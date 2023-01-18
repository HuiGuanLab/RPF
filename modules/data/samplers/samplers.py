import os
import math
import random
from torch.utils.data.sampler import Sampler

class TripletInfoNceSampler(Sampler):
    def __init__(self, cfg):
        self.num_triplets = cfg.DATA.NUM_TRIPLETS#100000=100k
        self.batch_size = cfg.DATA.TRAIN_BATCHSIZE#16
        self.attrs = cfg.DATA.ATTRIBUTES.NAME#sleeve length、lapel design、collar_design...
        self.num_values = cfg.DATA.ATTRIBUTES.NUM#[6,9,5,...]
        self.indices = {}#dictionary {attr:[0,1,2,3...numvalue-1]} for instance==>sleeve length:[0,1,2,3,4,5]
        for i, attr in enumerate(self.attrs):
            self.indices[attr] = [[] for _ in range(self.num_values[i])]

        label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, cfg.DATA.GROUNDTRUTH.TRAIN)
        assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
        with open(label_file, 'r') as f:
            for l in f:
                l = [int(i) for i in l.strip().split()]#[0,7,3] [index,]
                fid = l[0]
                attr_val = [(l[i], l[i+1]) for i in range(1, len(l), 2)]#i=1 i+1=2 attr_val=[(7,3)]
                for attr, val in attr_val:
                    self.indices[self.attrs[attr]][val].append(fid)#indices['sleeve legnth'][3].append[0] explain:label index=0 attr=sleeve legnth,child attr=3,label list append 0

    def __len__(self):
        return math.ceil(self.num_triplets / self.batch_size)#100k/16 共有多少个minibatch？1个minibatch 有16个三元组

    def __str__(self):
        return f"| Triplet Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

    def __iter__(self):
        sampled_attrs = random.choices(range(0, len(self.attrs)), k=self.num_triplets)#pick 100k attrs from (0,attr_num-1),[0,7,2,7,3...]
        #为每个minibatch sample 16个三元组(a:长袖,p：长袖,n：短袖)
        for i in range(self.__len__()):
            attrs = sampled_attrs[i*self.batch_size:(i+1)*self.batch_size]#sampled_attrs for i_th mini_batch

            anchors = []
            positives = []
            negatives = []
            for a in attrs:#a is a specific attr index in attrs such as sleeve length=7
                # Randomly sample two attribute values
                vp, vn = random.sample(range(self.num_values[a]), 2)#random sample 2 unique child attrs index from  sleeve length,vp as positive child attr index,vn as negtive child attr index
                # Randomly sample an anchor image and a positive image
                x, p = random.sample(self.indices[self.attrs[a]][vp], 2)#random sample 2 unique images from vp child attr index as x and positive
                # Randomly sample a negative image
                n = random.choice(self.indices[self.attrs[a]][vn])#random sample a image from vn child attr index as negtive
                anchors.append((x, a, (a,vp)))
                positives.append((p, a, (a,vp)))
                negatives.append((n, a, (a,vn)))
           
            yield anchors + positives + negatives


class ImageSampler(Sampler):
    def __init__(self, cfg, file):
        self.batch_size = cfg.DATA.TEST_BATCHSIZE#64

        label_file = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET, file)
        assert os.path.exists(label_file), f"Train label file {label_file} does not exist."
        self.labels = []
        with open(label_file, 'r') as f:
            for l in f:
                l = [int(i) for i in l.strip().split()]
                self.labels.append(tuple(l))

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __str__(self):
        return f"| Image Sampler | iters {self.__len__()} | batch size {self.batch_size}|"

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.labels[i*self.batch_size:(i+1)*self.batch_size]

