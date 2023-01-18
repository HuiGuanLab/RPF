
import time
import torch
from modules.utils.metric import AverageMeter


def do_train(cfg, model, data_loader, gt, lt, optimizer, intra_loss, inter_loss, device, logger, epoch):
    losses = AverageMeter()
    if lt is not None:
        glosses = AverageMeter()
        llosses = AverageMeter()
        interlosses = AverageMeter()
        
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        x, p, n, a, flag, deflag= batch
        #x is list
        n_data = len(x)
        a = a.to(device)
     
        flag = flag.to(device)
        deflag = deflag.to(device)
        
        gxx = torch.stack([gt(i) for i in x], dim=0).to(device) #gx[B,C,H,W]
        gpp = torch.stack([gt(i) for i in p], dim=0).to(device)
        gnn = torch.stack([gt(i) for i in n], dim=0).to(device)

        data_time.update(time.time() - end)

        # equal 0 means same attribute
        a_flag = torch.cat([a,a,a],dim=0)
        a_row = a_flag.unsqueeze(-1)
        a_column = a_flag.unsqueeze(0)
        a_flag = a_row - a_column
        a_flag = a_flag.masked_fill(a_flag != 0., float(-10000.0)) 

        if lt is None: #stage1
            gx, gx_attnmap = model(gxx, a, level='global')
            gp, gp_attnmap = model(gpp, a, level='global')
            gn, gn_attnmap = model(gnn, a, level='global')
            loss = cfg.SOLVER.REGION_WEIGHT * intra_loss(gx, gp, gn)#三元组损失

        else:#stage2
            gx, gx_hat, gx_attnmap = model(gxx, a, level='global')
            gp, gp_hat, gp_attnmap = model(gpp, a, level='global')
            gn, gn_hat, gn_attnmap = model(gnn, a, level='global')
            loss = cfg.SOLVER.REGION_WEIGHT * intra_loss(gx, gp, gn)#三元组损失
            glosses.update(loss.cpu().item(), n_data)

            gx_attnmap = gx_attnmap.cpu().detach().numpy()
            gp_attnmap = gp_attnmap.cpu().detach().numpy()
            gn_attnmap = gn_attnmap.cpu().detach().numpy()

            lx = torch.stack([lt(i, mask) for i, mask in zip(x, gx_attnmap)], dim=0).to(device)
            lp = torch.stack([lt(i, mask) for i, mask in zip(p, gp_attnmap)], dim=0).to(device)
            ln = torch.stack([lt(i, mask) for i, mask in zip(n, gn_attnmap)], dim=0).to(device)

            lx = model(lx, a, level='local')
            lp = model(lp, a, level='local')
            ln = model(ln, a, level='local')

            # local losses
            l = intra_loss(lx, lp, ln)
            llosses.update(cfg.SOLVER.PATCH_WEIGHT * l.cpu().item(), n_data)
            loss += cfg.SOLVER.PATCH_WEIGHT * l
            
            pp = torch.cat([gx,gp,gn],dim=0)
            xx = torch.cat([lx,lp,ln],dim=0)
            nn = torch.cat([gx_hat,gp_hat,gn_hat],dim=0)
                        
            interloss = cfg.SOLVER.INTER_WEIGHT * inter_loss(pp, xx, nn, flag, deflag, a_flag, T = cfg.SOLVER.TAU, alpha = cfg.SOLVER.ALPHA, type=1)

            interlosses.update(interloss.cpu().item(), n_data)
            loss +=  interloss

        losses.update(loss.cpu().item(), n_data)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


        local_log = (f"Regin-intra Loss: {glosses.val:.4f}({glosses.avg:.4f})\t"+\
                    f"Inter Loss: {interlosses.val:.4f}({interlosses.avg:.4f})\t"+\
                    f"Patch-intra Loss: {llosses.val:.4f}({llosses.avg:.4f})\t") if lt is not None else ""
        if idx % cfg.SOLVER.LOG_PERIOD == 0:
            logger.info(f"Train Epoch: [{epoch}][{idx}/{len(data_loader)}]\t"+
                        local_log+
                         f"Loss: {losses.val:.4f}({losses.avg:.4f})\t"+
                         f"Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})\t"+
                         f"Data Time: {data_time.val:.3f}({data_time.avg:.3f})")
            
    return (losses.avg, glosses.avg, llosses.avg, interlosses.avg) if lt is not None else losses.avg



