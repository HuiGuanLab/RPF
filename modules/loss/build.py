from ast import If
import torch
import torch.nn as nn


def build_loss(cfg):
    return TripletRankingLoss(cfg)


class TripletRankingLoss(nn.Module):
    def __init__(self, cfg):
        super(TripletRankingLoss, self).__init__()
        self.margin = cfg.SOLVER.MARGIN
        self.device = torch.device(cfg.DEVICE)
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, ref, pos, neg):
        x1 = nn.functional.cosine_similarity(ref, pos, dim=1)
        x2 = nn.functional.cosine_similarity(ref, neg, dim=1)
        target = torch.FloatTensor(ref.size(0)).fill_(1)
        target = target.to(self.device)
        loss = self.criterion(x1, x2, target)

        return loss

class DELoss(nn.Module):
    def __init__(self):
        super(DELoss, self).__init__()

    def forward(self, xx, pp, nen, flag, deflag, a_flag,T=0.07,alpha=48.,type=1):
        '''
        FLAG [B,B]
        '''
        xx = nn.functional.normalize(xx, dim=1) #[B,C]
        pp = nn.functional.normalize(pp, dim=1) #[B,C]
        nen = nn.functional.normalize(nen, dim=1) #[B,C]

        singlex = torch.matmul(xx, xx.t()) # B B one - more
        xp = torch.matmul(xx, pp.t()) # B B one - more
        
        pos1 = singlex * torch.eye(singlex.shape[0])[:,:].cuda()
        # print(nominator.shape)
        pos1 = pos1.sum(dim=1).unsqueeze(1)
        pos = xp + flag
        neg1 = xp + deflag
        neg = torch.matmul(xx, nen.t()) # B B
        neg = neg + a_flag # B B

        pos = pos/T
        neg1 = neg1/T
        neg = neg/T
       
        nominator = torch.logsumexp(pos, dim=1)
        # print(nominator.shape)
        if type==1:
            de1 = torch.exp(torch.cat((pos,neg1),dim=1))
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))
            
            # print(denominator.shape)
            # denominator = torch.cat((pos, neg1, neg), dim=1)
        elif type==2:
            de1 = torch.exp(pos)
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))            
        elif type==3:
            nominator = torch.logsumexp(pos1, dim=1)
            de1 = torch.exp(torch.cat((pos1,neg1),dim=1))
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))
        elif type==4:
            nominator = torch.logsumexp(pos1, dim=1)
            de1 = torch.exp(pos1)
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))

        # denominator = torch.cat((pos, neg), dim=1)
        # print(denominator.shape)
        # denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)
