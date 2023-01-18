import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import copy
import logging
import math
import logging

from .resnet import get_model
from os.path import join as pjoin
from timm.models.layers import trunc_normal_
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class RPF(nn.Module):
    def __init__(self, cfg):
        super(RPF, self).__init__()
        self.stage = 1 
        self.choices = nn.ModuleDict({
            'global': nn.ModuleDict({
                'attrnet': AttrEmbedding(cfg.DATA.NUM_ATTRIBUTES, cfg.MODEL.ATTRIBUTE.EMBED_SIZE),
                'basenet': get_model(cfg.MODEL.GLOBAL.BACKBONE.NAME, pretrained=True),
                'attnnet': AttnEmbedding(
                    cfg.MODEL.ATTRIBUTE.EMBED_SIZE,
                    cfg.MODEL.GLOBAL.BACKBONE.EMBED_SIZE,
                    cfg.MODEL.GLOBAL.ATTENTION.SPATIAL.COMMON_EMBED_SIZE,
                    cfg.MODEL.GLOBAL.ATTENTION.SPATIAL.REDUCTION_RATE,
                    cfg.MODEL.EMBED_SIZE
                )
            })
        })
        if cfg.MODEL.TRANSFORMER.ENABLE:
            self.stage = 2 
            self.choices.update(
                {
                    'local': nn.ModuleDict({
                        'basenet': Transformer(
                            cfg.MODEL.EMBED_SIZE,
                            cfg.MODEL.TRANSFORMER.PATCH_SIZE,
                            cfg.MODEL.TRANSFORMER.SPLIT,
                            cfg.MODEL.TRANSFORMER.NUM_LAYERS, 
                            cfg.MODEL.TRANSFORMER.EMBED_SIZE, 
                            cfg.MODEL.ATTRIBUTE.EMBED_SIZE,
                            cfg.MODEL.TRANSFORMER.SLIDE_STEP,
                            cfg.MODEL.TRANSFORMER.MLP_DIM,
                            cfg.MODEL.TRANSFORMER.DROPOUT_RATE,
                            cfg.MODEL.TRANSFORMER.NUM_HEADS,
                            cfg.MODEL.TRANSFORMER.ATT_DROPOUT_RATE,
                            cfg.MODEL.TRANSFORMER.REDUCTION_RATE,
                            cfg.INPUT.LOCAL_SIZE
                        )
            })})

    def forward(self, x, a, level='global'):
        a = self.choices['global']['attrnet'](a)
        if level=='global':
            x = self.choices["global"]['basenet'](x)
            #16 1024 14 14
            x, x_hat,attmap,_= self.choices["global"]['attnnet'](x, a)
            if self.stage == 2:
                return x, x_hat,attmap
            return x, attmap
        elif level=='local':
            x=self.choices["local"]['basenet'](x, a)
            return x

    def load_state_dict(self, loaded_state_dict):
        state = super(RPF, self).state_dict()
        for k in loaded_state_dict:
            if k in state:
                state[k] = loaded_state_dict[k]
        super(RPF, self).load_state_dict(state)

    def load_from(self, weights):
        with torch.no_grad():
            self.choices['local']['basenet'].embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.choices['local']['basenet'].embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.choices['local']['basenet'].embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                print("load_pretrained: posemb_new=posemb")
                logger.info("load_pretrained: posemb_new=posemb")
                self.choices['local']['basenet'].embeddings.position_embeddings.copy_(posemb)       
            for bname, block in self.choices['local']['basenet'].encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        # print(uname)
                        unit.load_from(weights, n_block=uname)

class Transformer(nn.Module):
    def __init__(self, final_embed_size, patch_size, split, num_layers, embed_size, attr_size, slide_step, mlp_dim, dropout_rate, num_heads, att_dropout_rate, reduction_rate,
    img_size=224):
        super(Transformer, self).__init__()
        # patch embeddings
        self.embeddings = ClsEmbeddings(patch_size, split, embed_size, slide_step,dropout_rate, img_size=img_size)
        # ViT backbone
        self.encoder = Encoder(num_layers, embed_size, mlp_dim, dropout_rate, num_heads, att_dropout_rate)
        # Use attribute-aware Transformer
        self.crossencoder=CrossEncoder(final_embed_size, embed_size, reduction_rate, dropout_rate, num_heads, att_dropout_rate)
        # Type embedding and initialization(there are two kinds of type embeddings)
        self.type_embed = nn.Parameter(torch.zeros(1, 2, embed_size))
        trunc_normal_(self.type_embed, std=.02)
       
        self.norm=LayerNorm(final_embed_size, eps=1e-6) 
        self.attr2query = Linear(attr_size, embed_size)
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, a):
        # attribute vector 512->768
        a = self.attr2query(a).unsqueeze(1)
        # Split the ROI into small patches
        x = self.embeddings(x)
        B, N, _ = x.shape
        # Using ViT encode the small patches
        x_output = self.encoder(x)
        # Type embedding
        a_type = self.type_embed[:,0]
        a = a + a_type
        x_type = self.type_embed[:,1].expand(B, N, -1)
        x_output = x_output + x_type
        # Using attribute-aware Transformer fliter out certain patches
        x_output=self.crossencoder(x_output, a).squeeze(dim=1)

        f = self.norm(x_output)

        return f
   
    
class Attention(nn.Module):
    def __init__(self, num_heads,embed_size,att_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads #12
        self.attention_head_size = int(embed_size / self.num_attention_heads)#64
        self.all_head_size = self.num_attention_heads * self.attention_head_size#64*12

        self.query = Linear(embed_size, self.all_head_size)
        self.key = Linear(embed_size, self.all_head_size)
        self.value = Linear(embed_size, self.all_head_size)

        self.out = Linear(embed_size, embed_size)
        self.attn_dropout = Dropout(att_dropout_rate)
        self.proj_dropout = Dropout(att_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights

class CrossAttention(nn.Module):
    def __init__(self, num_heads,embed_size,att_dropout_rate):
        super(CrossAttention, self).__init__()
        print("head:"+str(num_heads))
        self.num_attention_heads = num_heads #12
        self.attention_head_size = int(embed_size / self.num_attention_heads)#64
        self.all_head_size = self.num_attention_heads * self.attention_head_size#64*12

        self.query = Linear(embed_size, self.all_head_size)
        self.key = Linear(embed_size, self.all_head_size)
        self.value = Linear(embed_size, self.all_head_size)

        self.out = Linear(embed_size, embed_size)
        self.attn_dropout = Dropout(att_dropout_rate)
        self.proj_dropout = Dropout(att_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,attr_embeding):
        mixed_query_layer = self.query(attr_embeding)#TODO [B,1,dK]
        # print("qqqqqq",mixed_query_layer.shape)
        mixed_key_layer = self.key(hidden_states)#B,N,dk
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs#softmaxattention权值记录
        # print(weights.shape)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        # print(attention_output.shape)
        return attention_output, weights

class CrossMlp(nn.Module):
    def __init__(self,embed_size,ratio,out_size,dropout_rate):
        super(CrossMlp, self).__init__()
        self.fc1 = Linear(embed_size,embed_size//ratio)
        self.fc2 = Linear(embed_size//ratio,out_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Mlp(nn.Module):
    def __init__(self,embed_size,mlp_dim,dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(embed_size,mlp_dim)
        self.fc2 = Linear(mlp_dim,embed_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ClsEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patch_size, split, embed_size, slide_step,dorpout_rate, img_size, in_channels=3):
        super(ClsEmbeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        if split == 'non-overlap':
            self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=embed_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif split== 'overlap':
            print("overlap mode")
            self.n_patches = ((img_size[0] - patch_size[0]) // slide_step + 1) * ((img_size[1] - patch_size[1]) // slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=embed_size,
                                        kernel_size=patch_size,
                                        stride=(slide_step, slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches+1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.dropout = Dropout(dorpout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings



class Block(nn.Module):
    def __init__(self, embed_size,mlp_dim,dropout_rate,num_heads,att_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = embed_size
        self.attention_norm = LayerNorm(embed_size, eps=1e-6)
        self.ffn_norm = LayerNorm(embed_size, eps=1e-6)
        self.ffn = Mlp(embed_size,mlp_dim,dropout_rate)
        self.attn = Attention(num_heads,embed_size,att_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))



class Encoder(nn.Module):
    def __init__(self, num_layers,embed_size,mlp_dim,dropout_rate,num_heads,att_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(num_layers):
            layer = Block(embed_size,mlp_dim,dropout_rate,num_heads,att_dropout_rate)
            self.layer.append(copy.deepcopy(layer))
    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)   

        return hidden_states

class CrossEncoder(nn.Module):
    def __init__(self, final_embed_size, embed_size, reduction_rate, dropout_rate, num_heads, att_dropout_rate):
        super(CrossEncoder, self).__init__()
        self.hidden_size = embed_size
        self.cross_attention_norm = LayerNorm(embed_size, eps=1e-6)
        self.cross_attn = CrossAttention(num_heads,embed_size,att_dropout_rate)

        self.fc = Linear(embed_size, final_embed_size)
        self.ffn = CrossMlp(final_embed_size, reduction_rate, final_embed_size, dropout_rate)
        self.ffn_norm = LayerNorm(final_embed_size, eps=1e-6)


    def forward(self, x,a):
        # Discard the useless cla token
        x = x[:,1:] 
        # Layer Norm
        x = self.cross_attention_norm(x)
        a = self.cross_attention_norm(a)
        # Cross Attention
        feature, _ = self.cross_attn(x, a)
        # Map the vector into 1024
        feature =self.fc(feature).squeeze()
        # MLP
        h = feature
        feature = self.ffn_norm(feature)
        feature = self.ffn(feature)
        feature = feature + h

        return feature


class AttrEmbedding(nn.Module):
    def __init__(self, n_attrs, embed_size):
        super(AttrEmbedding, self).__init__()
        self.attr_embedding  = torch.nn.Embedding(n_attrs, embed_size)

    def forward(self, x):
        return self.attr_embedding(x)#one-hot (nx1)->embed (512x1) n is total number of different attributes(such as sleeve length/lapel design)

class RegionMlp(nn.Module):
    def __init__(self,embed_size,mlp_dim,dropout_rate=0.1):
        super(RegionMlp, self).__init__()
        self.fc1 = nn.Linear(embed_size,mlp_dim)
        self.fc2 = nn.Linear(mlp_dim,embed_size)
        self.act_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class AttnEmbedding(nn.Module):
    def __init__(
        self, 
        attr_embed_size, 
        img_embed_size, 
        common_embed_size, 
        reduction_rate, 
        final_embed_size):
        super(AttnEmbedding, self).__init__()
        
        self.attr_transform1 = nn.Linear(
            attr_embed_size, 
            common_embed_size
        )
        self.conv = nn.Conv2d(
            img_embed_size, 
            common_embed_size, 
            kernel_size=1, stride=1
        )
        self.feature_fc = nn.Linear(
            img_embed_size,
            final_embed_size
        )
        self.x_layer_norm = nn.LayerNorm(img_embed_size, eps=1e-6)
        self.a_layer_norm = nn.LayerNorm(attr_embed_size, eps=1e-6)
        self.mlp_layer_norm = nn.LayerNorm(final_embed_size, eps=1e-6)
        self.mlp = RegionMlp(final_embed_size, final_embed_size//reduction_rate)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.aapool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x, a):
        B,C,H,W = x.shape
        ###############################
        x = self.x_layer_norm(x.reshape(B,C,-1).permute(0,2,1)).permute(0,2,1).reshape(B,C,H,W)
        a = self.a_layer_norm(a)
        ###############################
        attmap, attmap_hat = self.spatial_attn(x, a)
        x_hat = x * attmap_hat            
        x = x * attmap

        x = x.view(x.size(0), x.size(1), -1)
        x_hat = x_hat.view(x_hat.size(0), x_hat.size(1), -1)
        
        x = x.sum(dim=2)
        x_hat = x_hat.sum(dim=2)

        x = self.feature_fc(x)
        ### mlp ###
        h = x
        x = self.mlp_layer_norm(x)
        x = self.mlp(x)
        x = x + h
        x = self.mlp_layer_norm(x)
                        
        ###x_hat use share mlp###
        g = x_hat
        x_hat = self.mlp_layer_norm(x_hat)
        x_hat = self.mlp(x_hat)
        x_hat = x_hat + g
        x_hat = self.mlp_layer_norm(x_hat)

        return x, x_hat, attmap.squeeze() , attmap_hat.squeeze()

    def spatial_attn(self, x, a):	
        x = self.conv(x)
        x = self.tanh(x)

        a = self.attr_transform1(a)
        a = self.tanh(a)

        a = a.view(a.size(0), a.size(1), 1, 1)
        a = a.expand_as(x)

        attmap = a * x
        attmap = torch.sum(attmap, dim=1, keepdim=True)
        attmap = torch.div(attmap, x.size(1) ** 0.5)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)

        attmap_hat = 1. - attmap
        attmap_hat = torch.div(attmap_hat,torch.sum(attmap_hat,dim=-1).unsqueeze(1))
       
        attmap = attmap.view(attmap.size(0), attmap.size(1), x.size(2), x.size(3))
        attmap_hat = attmap_hat.view(attmap.size(0), attmap.size(1), x.size(2), x.size(3))

        return attmap, attmap_hat