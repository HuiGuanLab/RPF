import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import itertools
def _image_reader(path):
    return Image.open(path).convert('RGB')


class BaseDataSet(Dataset):
    def __init__(self, cfg, split):
        self.root_path = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET)#/home/data/datasets/FGdatas/FashionAI

        self.fnamelist = []
        filepath = os.path.join(self.root_path, cfg.DATA.PATH_FILE[split])# /home/data/datasets/FGdatas/FashionAI/filenames_train.txt(if split='TRAIN')
        assert os.path.exists(filepath), f"File {filepath} does not exist."
        with open(filepath, 'r') as f:
            for l in f:
                self.fnamelist.append(l.strip())#each image in filenames_train.txt append to fnamelist one by one(if split='TRAIN')
        
        self.image_loader = _image_reader

    def __len__(self):
        return self.fnamelist

    def __getitem__(self, index):
        # print(index)
        path = os.path.join(self.root_path, self.fnamelist[index[0]])
        assert os.path.exists(path), f"File {path} does not exist."

        img = self.image_loader(path)

        return (img,) + index[1:]

def tripletInfo_collate_fn(batch):
    xpn= batch
    n = len(xpn) // 3
    x, x_a, avx = zip(*xpn[:n])
    p, p_a, avp = zip(*xpn[n:2*n])
    n, n_a, avn = zip(*xpn[2*n:3*n])
    av = avx + avp + avn

    flag = [torch.tensor(0) if x[0]==x[1] else torch.tensor(-10000.0) for x in itertools.product(av, av)]
    flag = torch.stack(flag,dim=0)
    
    de_flag = [torch.tensor(-10000.0) if x[0]==x[1] else torch.tensor(0) for x in itertools.product(av, av)]
    de_flag = torch.stack(de_flag,dim=0)

    flag = flag.reshape(len(av), -1)
    de_flag = de_flag.reshape(len(av), -1)

    return x, p, n, torch.LongTensor(x_a), flag, de_flag

def image_collate_fn(batch):
    # print(batch)
    x, a, v = zip(*batch)

    a = torch.LongTensor(a)
    v = torch.LongTensor(v)

    return x, a, v
