import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2,5,6,7,8'

import sys
sys.path.append('../Pedestrain_intent_v3')

from torch import nn, optim
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import argparse
from configs import cfg

from datasets import make_dataloader
from lib.modeling import make_model
from lib.engine.trainer import do_train, do_val, do_train_iteration, do_train_iteration_intent
from lib.utils.scheduler import ParamScheduler, sigmoid_anneal

import logging
from termcolor import colored 
from lib.utils.meter import AverageValueMeter
from lib.engine.inference import inference, inference_intent

from torch.nn.parallel import DataParallel
import utils


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--seed', default=0, type=int)   
    # 添加分布式训练的参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()

    

def main(args):
    utils.init_distributed_mode(args)
    
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)  
    # np.random.seed(seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    cudnn.benchmark = True

    # num_tasks = utils.get_world_size()  # 用于获取当前分布式计算节点的数量（即计算节点的数量）
    # global_rank = utils.get_rank()      # 用于获取当前计算节点在分布式训练中的排名（即节点的索引）
    # sampler_rank = global_rank    
    
  
    logger = logging.Logger("action_intent")
    run_id = 'no_wandb'

    # make dataloader
    train_dataloader = make_dataloader(cfg, split='train', distributed=True)
    val_dataloader = make_dataloader(cfg, split='val', distributed=True)
    test_dataloader = make_dataloader(cfg, split='test', distributed=True)

    # make model
    model = make_model(cfg).to(cfg.DEVICE) # DDP
    
    # model = DataParallel(model) # DP


    # 分布计算
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    num_params = 0
    for name, param in model.named_parameters():
        _num = 1
        for a in param.shape:
            _num *= a
        num_params += _num
        print("{}:{}".format(name, param.shape))
    print(colored("total number of parameters: {}".format(num_params), 'white', 'on_green'))

    optimizer = optim.RMSprop(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.L2_WEIGHT, alpha=0.9, eps=1e-7)# the weight of L2 regularizer is 0.001
    # optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate.base, weight_decay=cfg.weight_decay, betas=cfg.optimizer.betas, eps=1e-8)# the weight of L2 regularizer is 0.001
    if cfg.SOLVER.SCHEDULER == 'exp':
        # NOTE: June 10, think about using Trajectron++ shceduler
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.SCHEDULER == 'plateau':
        # Same to original PIE implementation
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10,#0.2
                                                            min_lr=1e-07, verbose=1)
    else:
        lr_scheduler = None #optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)
        
    # checkpoints
    if os.path.isfile(cfg.CKPT_DIR):
        model.load_state_dict(torch.load(cfg.CKPT_DIR))
        save_checkpoint_dir = os.path.join('/'.join(cfg.CKPT_DIR.split('/')[:-2]), run_id)
        print(colored("Train from checkpoint: {}".format(cfg.CKPT_DIR), 'white', 'on_green'))
    else:
        save_checkpoint_dir = os.path.join(cfg.CKPT_DIR, run_id)
    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
        
    # NOTE: Setup parameter scheduler
    if cfg.SOLVER.INTENT_WEIGHT_MAX != -1:
        model.param_scheduler = ParamScheduler()
        model.param_scheduler.create_new_scheduler(
                                            name='intent_weight',
                                            annealer=sigmoid_anneal,
                                            annealer_kws={
                                                'device': cfg.DEVICE,
                                                'start': 0,
                                                'finish': cfg.SOLVER.INTENT_WEIGHT_MAX,# 20.0
                                                'center_step': cfg.SOLVER.CENTER_STEP,#800.0,
                                                'steps_lo_to_hi': cfg.SOLVER.STEPS_LO_TO_HI, #800.0 / 4.
                                            })
    torch.autograd.set_detect_anomaly(True)

    # train
    loss_act_det_meter = AverageValueMeter()
    loss_act_pred_meter = AverageValueMeter()
    loss_intent_meter = AverageValueMeter()
    
    # # 混合精度训练
    # scaler = torch.cuda.amp.GradScaler()

    if cfg.DATALOADER.ITERATION_BASED:
        do_train_iteration_intent(
            cfg, model, optimizer, # scaler,
            train_dataloader, val_dataloader, test_dataloader, 
            cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler, save_checkpoint_dir=save_checkpoint_dir
        )
    else:
        loss_act_det_meter = AverageValueMeter()
        # loss_act_pred_meter = AverageValueMeter()
        loss_intent_meter = AverageValueMeter()
        print("epoch train!")
        for epoch in range(cfg.SOLVER.MAX_EPOCH):
            if args.distributed:
                train_dataloader.sampler.set_epoch(epoch)           
            # inference_intent(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger) # debug用
            do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.DEVICE, loss_act_det_meter, loss_act_pred_meter, loss_intent_meter, logger=logger, lr_scheduler=lr_scheduler)
            loss_val = do_val(cfg, epoch, model, val_dataloader, cfg.DEVICE, logger=logger)

            if epoch % cfg.TEST.INTERVAL == 0:
                result_dict = inference_intent(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger)
                if 'intent' in cfg.MODEL.TASK:
                    save_file = os.path.join(save_checkpoint_dir, 
                                        'iters_{}_acc_{:.3}_f1_{:.3}.pth'.format(str(epoch).zfill(3), 
                                                                            result_dict['intent_accuracy'],
                                                                            result_dict['intent_f1']))
                else:
                    save_file = os.path.join(save_checkpoint_dir, 
                                    'iters_{}_mAP_{:.3}.pth'.format(str(epoch).zfill(3), 
                                                                        result_dict['mAP']))
                torch.save(model.state_dict(), save_file)
                # torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))
            if cfg.SOLVER.SCHEDULER == 'plateau':
                lr_scheduler.step(loss_val)
                
            
if __name__ == '__main__':
    args = get_args()
    main(args)