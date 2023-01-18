from unittest import result
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from modules.utils.metric import APScorer, AverageMeter


def do_eval(
    model,
    query_loader,
    candidate_loader,
    gt,
    lt,
    attrs,
    device,
    logger,
    epoch=-1,
    beta=0.6
):
    mAPs = AverageMeter()
    mAPs_g = AverageMeter()
    mAPs_l = AverageMeter()


    logger.info("Begin evaluation.")
    model.eval()

    logger.info("Forwarding query images...")
    q_feats,q_values = extract_features(model, query_loader, gt, lt, device, len(attrs), beta=beta)
    logger.info("Forwarding candidate images...")
    c_feats,c_values = extract_features(model, candidate_loader, gt, lt, device, len(attrs), beta=beta)
    if lt is not None:
        q_feats_g=q_feats[1]
        q_feats_l=q_feats[2]
        q_feats=q_feats[0]

        c_feats_g=c_feats[1]
        c_feats_l=c_feats[2]
        c_feats=c_feats[0]

    for i, attr in enumerate(attrs):
        mAP = mean_average_precision(q_feats[i], c_feats[i], q_values[i], c_values[i])
        logger.info(f"{attr} MeanAP: {100.*mAP:.4f}")
        mAPs.update(mAP, q_feats[i].shape[0])

        if lt is not None:
            mAP_g= mean_average_precision(q_feats_g[i], c_feats_g[i], q_values[i], c_values[i])
            mAP_l= mean_average_precision(q_feats_l[i], c_feats_l[i], q_values[i], c_values[i])
            logger.info(f"{attr} MeanAP_r: {100.*mAP_g:.4f}")
            logger.info(f"{attr} MeanAP_p: {100.*mAP_l:.4f}")
            mAPs_g.update(mAP_g, q_feats[i].shape[0])
            mAPs_l.update(mAP_l, q_feats[i].shape[0])


    local_log = (f"Total MeanAP_r: {100.*mAPs_g.avg:.4f}\t"+\
                 f"Total MeanAP_p: {100.*mAPs_l.avg:.4f}\t") if lt is not None else ""
    logger.info(f"Total MeanAP: {100.*mAPs.avg:.4f}\t"+
                local_log)
    
    return (mAPs.avg,mAPs_g.avg,mAPs_l.avg) if lt is not None else mAPs.avg


def extract_features(model, data_loader, gt, lt, device, n_attrs, beta=0.6):
    feats = []
    feats_g= []
    feats_l= []
    indices = [[] for _ in range(n_attrs)]
    values = []
    with tqdm(total=len(data_loader)) as bar:
        cnt = 0
        for idx, batch in enumerate(data_loader):
            x, a, v = batch#x=index of 
            # print("eval a shape",a.shape)
            a = a.to(device)
            
            out= process_batch(model, x, a, gt, lt, device, beta=beta)
            if lt is not None:
                out_g=out[1]
                out_l=out[2]
                out=out[0]
            
            feats.append(out.cpu().numpy())
            values.append(v.numpy())
            if lt is not None:
                feats_g.append(out_g.cpu().numpy())
                feats_l.append(out_l.cpu().numpy())

            for i in range(a.size(0)):
                indices[a[i].cpu().item()].append(cnt)
                cnt += 1

            bar.update(1)
    feats = np.concatenate(feats)
    values = np.concatenate(values)
    feats = [feats[indices[i]] for i in range(n_attrs)]
    values = [values[indices[i]] for i in range(n_attrs)]

    if lt is not None:
        feats_g = np.concatenate(feats_g)
        feats_l = np.concatenate(feats_l)
        feats_g = [feats_g[indices[i]] for i in range(n_attrs)]
        feats_l = [feats_l[indices[i]] for i in range(n_attrs)]
      
        return (feats,feats_g,feats_l) ,values


    return feats,values


def process_batch(model, x, a, gt, lt, device, beta=0.6):
    gx = torch.stack([gt(i) for i in x], dim=0)
    gx = gx.to(device)
    with torch.no_grad():
        if lt is not None:
            g_feats, _, attmap = model(gx, a, level='global')
        else:
            g_feats, attmap = model(gx, a, level='global')
    if lt is None:
        return nn.functional.normalize(g_feats, p=2, dim=1)

    attmap = attmap.cpu().numpy()

    lx = torch.stack([lt(i, mask) for i, mask in zip(x, attmap)], dim=0)
    lx = lx.to(device)
    with torch.no_grad():
        l_feats = model(lx, a, level='local')
    
    out = torch.cat((torch.sqrt(torch.tensor(beta)) * nn.functional.normalize(g_feats, p=2, dim=1),
            torch.sqrt(torch.tensor(1-beta)) * nn.functional.normalize(l_feats, p=2, dim=1)), dim=1)


    out_g=nn.functional.normalize(g_feats, p=2, dim=1)
    out_l=nn.functional.normalize(l_feats, p=2, dim=1)

    return (out,out_g,out_l)


def mean_average_precision(queries, candidates, q_values, c_values):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''
    
    scorer = APScorer(candidates.shape[0])#k=c
    # similarity matrix
    simmat = np.matmul(queries, candidates.T)#qxc

    ap_sum = 0
    for q in range(simmat.shape[0]):#第q行
        sim = simmat[q]#去相似矩阵的第q行
        index = np.argsort(sim)[::-1]#从小到大排序选最后一个最大的(从大到小排序)
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_values[index[i]] == q_values[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP