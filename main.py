import argparse
import os, sys
import time
import datetime

import torch

import numpy as np
from modules.config import cfg
from modules.utils.logger import setup_logger
from modules.data import build_data
from modules.model import build_model
from modules.solver import build_optimizer, build_lr_scheduler
from modules.data.transforms import GlobalTransform, LocalTransform
from modules.engine import do_eval, do_train
from modules.loss import build_loss
from modules.loss.build import DELoss
from modules.utils.checkpoint import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import random


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(cfg):
    logger = setup_logger(name=cfg.NAME, level=cfg.LOGGER.LEVEL, stream=cfg.LOGGER.STREAM)
    logger.info(cfg)
    device = torch.device(cfg.DEVICE)

    model = build_model(cfg)
    
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    logger.info(f"Number of parameters: {n_parameters}")
    
    model.to(device)

    start_epoch=0
    gt = GlobalTransform(cfg)#global stream data process
   
    if args.test is None:#prepare data
        train_loader, valid_query_loader, valid_candidate_loader = build_data(cfg)
    else:
        test_query_loader, test_candidate_loader = build_data(cfg, args.test)

    
    if args.resume is not None:
        path = args.resume
        if os.path.isfile(path):
            logger.info(f"Loading checkpoint '{path}'.")
            checkpoint = torch.load(path, map_location='cpu')
            logger.info(f"Best performance {checkpoint['mAP']} at epoch {checkpoint['epoch']}.")            
            logger.info(f"start at epoch {checkpoint['epoch']}.")
            model.load_state_dict(checkpoint['model'])
            logger.info(f"Loaded checkpoint '{path}'")
        else:
            logger.info(f"No checkpoint found at '{path}'.")
            sys.exit()
    if cfg.MODEL.TRANSFORMER.ENABLE:
        lt = LocalTransform(cfg)
        print("vit")
        if args.test is None:
            model.load_from(np.load("Your Downloaded Vit Pretrained Model"))


    if args.test is not None:
        logger.info(f"Begin test on {args.test} set.")
        do_eval(
            model, 
            test_query_loader, 
            test_candidate_loader, 
            gt, 
            lt if cfg.MODEL.TRANSFORMER.ENABLE else None, 
            cfg.DATA.ATTRIBUTES.NAME, 
            device, 
            logger, 
            epoch=-1, 
            beta=cfg.SOLVER.BETA
        )
        sys.exit()

    optimizer= build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    intra_criterion = build_loss(cfg)
    inter_criterion = DELoss().cuda()
  


    best_mAP = 0
    start = time.time()
    tbwriter = SummaryWriter('tensorboard/'+ cfg.NAME)

    for epoch in range(start_epoch,cfg.SOLVER.EPOCHS):
        logger.info(f"Region branch learning rate: {optimizer.param_groups[0]['lr']}.")
        if cfg.MODEL.TRANSFORMER.ENABLE:
            logger.info(f"Patch branch learning rate: {optimizer.param_groups[1]['lr']}.")

        losses = do_train(
            cfg, 
            model, 
            train_loader, 
            gt, 
            lt if cfg.MODEL.TRANSFORMER.ENABLE else None, 
            optimizer,
            intra_criterion, 
            inter_criterion, 
            device, 
            logger,
            epoch+1
        )
        if cfg.MODEL.SINGLE.ENABLE:
            tbwriter.add_scalar('Train/mean loss Total', losses, epoch+1)
        
        if cfg.MODEL.TRANSFORMER.ENABLE:
            tbwriter.add_scalar('Train/mean loss Total', losses[0], epoch+1)
            tbwriter.add_scalar('Train/Tri-loss r', losses[1], epoch+1)
            tbwriter.add_scalar('Train/Tri-loss p', losses[2], epoch+1)
            tbwriter.add_scalar('Train/Inter-loss', losses[3], epoch+1)


        if (epoch+1) % cfg.SOLVER.EVAL_STEPS == 0:
            logger.info("Lets go to test")
            mAP = do_eval(
                    model, 
                    valid_query_loader,
                    valid_candidate_loader, 
                    gt, 
                    lt if cfg.MODEL.TRANSFORMER.ENABLE else None, 
                    cfg.DATA.ATTRIBUTES.NAME, 
                    device, 
                    logger, 
                    epoch=epoch+1, 
                    beta=cfg.SOLVER.BETA
                    )
            if  cfg.MODEL.TRANSFORMER.ENABLE:
                mAP_g = mAP[1]
                mAP_l = mAP[2]
                mAP = mAP[0]
                tbwriter.add_scalar('Test/mean AP_r', mAP_g, epoch+1)
                tbwriter.add_scalar('Test/mean AP_p', mAP_l, epoch+1)

            tbwriter.add_scalar('Test/mean AP', mAP, epoch+1)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'epoch': epoch+1,
                'model': model.state_dict(),
                'mAP': mAP
            }, is_best, path=os.path.join(cfg.SAVE_DIR, cfg.NAME))
        scheduler.step()
        


    end = time.time()
    total_time = end - start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total training time: {total_time_str}')
    tbwriter.close()#关闭writer

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Attribute Specific Embedding Network")
    parser.add_argument(
        "--cfg", nargs="+", help="config file", default=None, type=str
    )
    parser.add_argument(
        "--test", help="run test on validation or test set", default=None, type=str
    )
    parser.add_argument(
        "--resume", help="checkpoint model to resume", default=None, type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)#solve the cpu problem
    args = parse_args()
    if args.cfg is not None:
        for cfg_file in args.cfg:
            cfg.merge_from_file(cfg_file)
    cfg.freeze()
    set_seed()
    main(cfg)
