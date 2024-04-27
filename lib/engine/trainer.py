import os
import numpy as np
import torch
import torch.nn.functional as F
from lib.utils.visualization import Visualizer, vis_results, print_info
from lib.modeling.layers.cls_loss import binary_cross_entropy_loss, cross_entropy_loss, trn_loss
from lib.utils.meter import AverageValueMeter
from .inference import inference, inference_intent
from tqdm import tqdm
import time

# 混合精度
from torch.cuda.amp import autocast as autocast

def do_train(cfg, 
             epoch, 
             model, 
             optimizer, 
             dataloader, 
             device, 
             loss_act_det_meter, 
             loss_act_pred_meter, 
             loss_intent_meter, 
             logger=None, 
             lr_scheduler=None):
    
    model.train()
    max_iters = len(dataloader)
    viz = Visualizer(cfg, mode='image')

    loss_func = {}
    loss_func['act_det'] = binary_cross_entropy_loss if cfg.MODEL.ACTION_LOSS == 'bce' else cross_entropy_loss
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss
    with torch.set_grad_enabled(True):
        end = time.time()
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            data_time = time.time() - end
            imgs = batch['crop_imgs'].to(device)
            # pedes_imgs = batch['pedes_imgs'].to(device)
            target_intent = batch['obs_intent'].to(device)
            target_action = batch['obs_action'].to(device)
            # img_path = batch['cur_image_file']
            
            # img_full = batch['full_imgs'].to(device) # 全景图片
            
            # # 只使用crop_imgs            
            # int_det_scores, act_det_scores = model(imgs, False)
            
            # withtraffic
            bboxes = batch['obs_bboxes'].to(device)
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO or cfg.MODEL.WITH_TRAFFIC else None
            x_traffic = None
            
            if cfg.MODEL.WITH_TRAFFIC:
                if cfg.MODEL.PRETRAINED:
                    x_traffic = batch['traffic_features'].to(device)
                else:
                    x_traffic = {}
                    if 'x_neighbor' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_neighbor'] = batch['neighbor_bboxes']
                        x_traffic['cls_neighbor'] = batch['neighbor_classes']

                        
                    if 'x_light' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_light'] = batch['traffic_light']
                        x_traffic['cls_light'] = batch['traffic_light_classes']

                        
                    if 'x_sign' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_sign'] = batch['traffic_sign']
                        x_traffic['cls_sign'] = batch['traffic_sign_classes']

                        
                    if 'x_crosswalk' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_crosswalk'] = batch['crosswalk']
                        x_traffic['cls_crosswalk'] = batch['crosswalk_classes']

                       
                    if 'x_station' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_station'] = batch['station']
                        x_traffic['cls_station'] = batch['station_classes']  
                        
                        
            # # 混合精度训练
            # with autocast():
            int_det_scores, act_det_scores = model(x=imgs,
                                                x_bbox=bboxes,
                                                x_ego=ego_motion,
                                                x_traffic=x_traffic)
            # get loss and update loss meters
            loss, loss_dict = 0.0, {}
            # action detection loss
            if act_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_act_det = loss_func['act_det'](act_det_scores, target_action)
                elif cfg.STYLE == 'SF-GRU':
                    loss_act_det = loss_func['act_det'](act_det_scores[:, -1:], target_action[:, -1:])
                loss += loss_act_det
                loss_act_det_meter.add(loss_act_det.item())
                loss_dict['loss_act_det_train'] = loss_act_det_meter.mean
            # intent loss
            if int_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_intent = loss_func['int_det'](int_det_scores, target_intent)
                elif cfg.STYLE == 'SF-GRU':
                    loss_intent = loss_func['int_det'](int_det_scores[:, -1:], target_intent[:, -1:])
                if act_det_scores is not None and hasattr(model, 'param_scheduler'):
                    loss += model.param_scheduler.intent_weight * loss_intent
                    loss_dict['intent_weight'] = model.param_scheduler.intent_weight.item()
                else:
                    loss += loss_intent
                loss_intent_meter.add(loss_intent.item())
                loss_dict['loss_int_det_train'] = loss_intent_meter.mean
            
            # weight
            if hasattr(model, 'param_scheduler'):  # intent_weight就是在计算intent loss时，乘的权重，也就是说在训练过程中，intent loss最初权重是很小的，然后逐渐增大
                model.param_scheduler.step()

            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # gradient clip
            loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            batch_time = time.time() - end
            loss_dict['batch_time'] = batch_time
            loss_dict['data_time'] = data_time

            if cfg.SOLVER.SCHEDULER == 'exp':
                lr_scheduler.step()
            # print log
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, loss_dict, optimizer=optimizer, logger=logger)
            # visualize
            # if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0 and hasattr(logger, 'log_image'):
            #     bboxes = bboxes.detach().cpu().numpy()
            #     if cfg.DATASET.BBOX_NORMALIZE:
            #         # NOTE: denormalize bboxes
            #         _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
            #         _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
            #         bboxes = bboxes * (_max - _min) + _min

            #     id_to_show = np.random.randint(bboxes.shape[0])
            #     gt_behaviors, pred_behaviors = {}, {}
            #     if 'action' in cfg.MODEL.TASK:
            #         target_action = target_action.detach().cpu().numpy()
            #         if act_det_scores.shape[-1] == 1:
            #             act_det_scores = act_det_scores.sigmoid().detach().cpu().numpy()
            #         else:
            #             act_det_scores = act_det_scores.softmax(dim=-1).detach().cpu().numpy()
            #         gt_behaviors['action'] = target_action[id_to_show, -1]
            #         pred_behaviors['action'] = act_det_scores[id_to_show, -1]

            #     if 'intent' in cfg.MODEL.TASK:
            #         target_intent = target_intent.detach().cpu().numpy()
            #         if int_det_scores.shape[-1] == 1:
            #             int_det_scores = int_det_scores.sigmoid().detach().cpu().numpy()
            #         else:
            #             int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy() 
            #         gt_behaviors['intent'] = target_intent[id_to_show, -1]
            #         pred_behaviors['intent'] = int_det_scores[id_to_show, -1]

            #     # visualize result  
            #     vis_results(viz, 
            #                 img_path[id_to_show], 
            #                 bboxes[id_to_show][-1], 
            #                 gt_behaviors=gt_behaviors,
            #                 pred_behaviors=pred_behaviors,
            #                 name='intent_train',
            #                 logger=logger)
                
            end = time.time()
    
def do_val(cfg, epoch, model, dataloader, device, logger=None, iteration_based=False):
    model.eval()
    loss_act_det_meter = AverageValueMeter()
    loss_act_pred_meter = AverageValueMeter()
    loss_intent_meter = AverageValueMeter()

    loss_func = {}
    loss_func['act_det'] = binary_cross_entropy_loss if cfg.MODEL.ACTION_LOSS == 'bce' else cross_entropy_loss
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss

    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            imgs = batch['crop_imgs'].to(device)
            # pedes_imgs = batch['pedes_imgs'].to(device)
            target_intent = batch['obs_intent'].to(device)
            target_action = batch['obs_action'].to(device)
            pose = batch['pose'].to(device)            
            # int_det_scores, act_det_scores = model(imgs, False)
            
            # withtraffic
            bboxes = batch['obs_bboxes'].to(device)
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO or cfg.MODEL.WITH_TRAFFIC else None
            x_traffic = None
            
            # img_full = batch['full_imgs']
            
            if cfg.MODEL.WITH_TRAFFIC:
                if cfg.MODEL.PRETRAINED:
                    x_traffic = batch['traffic_features'].to(device)
                else:
                    x_traffic = {}
                    if 'x_neighbor' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_neighbor'] = batch['neighbor_bboxes']
                        x_traffic['cls_neighbor'] = batch['neighbor_classes']
                        
                    if 'x_light' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_light'] = batch['traffic_light']
                        x_traffic['cls_light'] = batch['traffic_light_classes']
                        
                    if 'x_sign' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_sign'] = batch['traffic_sign']
                        x_traffic['cls_sign'] = batch['traffic_sign_classes']
                        
                    if 'x_crosswalk' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_crosswalk'] = batch['crosswalk']
                        x_traffic['cls_crosswalk'] = batch['crosswalk_classes']
                       
                    if 'x_station' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_station'] = batch['station']
                        x_traffic['cls_station'] = batch['station_classes']            
            
            # with autocast():
            int_det_scores, act_det_scores = model(x=imgs,
                                                x_pose=pose,
                                                x_bbox=bboxes,
                                                x_ego=ego_motion,
                                                x_traffic=x_traffic)
            
            if act_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_act_det_meter.add(loss_func['act_det'](act_det_scores, target_action).item())
                elif cfg.STYLE == 'SF-GRU':
                    # loss_act_det_meter.add(loss_func['act_det'](act_det_scores[:, -1:], target_action[:, -1:]).item())
                    loss_act_det_meter.add(loss_func['act_det'](act_det_scores, target_action[:, -1:]).item())
                
            if int_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_intent_meter.add(loss_func['int_det'](int_det_scores, target_intent).item())
                elif cfg.STYLE == 'SF-GRU':
                    # loss_intent_meter.add(loss_func['int_det'](int_det_scores[:, -1:], target_intent[:, -1:]).item())    
                    loss_intent_meter.add(loss_func['int_det'](int_det_scores, target_intent[:, -1:]).item())    

    loss_dict = {}
    if 'action' in cfg.MODEL.TASK:
        loss_dict['loss_act_det_val'] = loss_act_det_meter.mean
        if 'trn' in cfg.MODEL.ACTION_NET:
            loss_dict['loss_act_pred_val'] = loss_act_pred_meter.mean
    if 'intent' in cfg.MODEL.TASK:
        loss_dict['loss_intent_val'] = loss_intent_meter.mean
    print_info(epoch, model, loss_dict, optimizer=None, logger=logger, iteration_based=iteration_based)
    
    return sum([v for v in loss_dict.values()])

def do_train_iteration(cfg, model, optimizer, 
                       train_dataloader, val_dataloader, test_dataloader, 
                       device, logger=None, lr_scheduler=None, save_checkpoint_dir=None):
    model.train()
    max_iters = len(train_dataloader)
    viz = Visualizer(cfg, mode='image')
    # trainning loss meters
    loss_act_det_meter = AverageValueMeter()
    loss_act_pred_meter = AverageValueMeter()
    loss_intent_meter = AverageValueMeter()
    # loss functions
    loss_func = {}
    loss_func['act_det'] = binary_cross_entropy_loss if cfg.MODEL.ACTION_LOSS == 'bce' else cross_entropy_loss
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss
    loss_func['act_pred'] = trn_loss if 'trn' in cfg.MODEL.ACTION_NET else None
    with torch.set_grad_enabled(True):
        end = time.time()
        # result_dict = inference(cfg, 100, model, test_dataloader, device, logger=logger, iteration_based=True) # 为了测试 
        for iters, batch in enumerate(tqdm(train_dataloader), start=1):
            data_time = time.time() - end

            x = batch['img_patches'].to(device) if batch['img_patches'] is not None else None
            bboxes = batch['obs_bboxes'].to(device)
            local_bboxes = batch['local_bboxes'].to(device) if batch['local_bboxes'] is not None else None
            masks = None#batch['masks'].to(device)
            img_path = batch['cur_image_file']
            target_intent = batch['obs_intent'].to(device)
            target_action = batch['obs_action'].to(device)
            target_future_action = batch['pred_action'].to(device)
            
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO or cfg.MODEL.WITH_TRAFFIC else None
            x_traffic = None
            
            if cfg.MODEL.WITH_TRAFFIC:
                if cfg.MODEL.PRETRAINED:
                    x_traffic = batch['traffic_features'].to(device)
                else:
                    x_traffic = {}
                    if 'x_neighbor' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_neighbor'] = batch['neighbor_bboxes']
                        x_traffic['cls_neighbor'] = batch['neighbor_classes']
                        
                    if 'x_light' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_light'] = batch['traffic_light']
                        x_traffic['cls_light'] = batch['traffic_light_classes']
                        
                    if 'x_sign' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_sign'] = batch['traffic_sign']
                        x_traffic['cls_sign'] = batch['traffic_sign_classes']
                        
                    if 'x_crosswalk' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_crosswalk'] = batch['crosswalk']
                        x_traffic['cls_crosswalk'] = batch['crosswalk_classes']
                       
                    if 'x_station' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_station'] = batch['station']
                        x_traffic['cls_station'] = batch['station_classes']
             
                     
            act_det_scores, act_pred_scores, int_det_scores, _ = model(x, 
                                                                                bboxes, 
                                                                                x_ego=ego_motion, 
                                                                                x_traffic=x_traffic,
                                                                                local_bboxes=local_bboxes, 
                                                                                masks=masks)
            # get loss and update loss meters
            # action detection loss
            loss, loss_dict = 0.0, {}
            if act_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_act_det = loss_func['act_det'](act_det_scores, target_action)
                elif cfg.STYLE == 'SF-GRU':
                    loss_act_det = loss_func['act_det'](act_det_scores[:, -1:], target_action[:, -1:])
                loss += loss_act_det
                loss_act_det_meter.add(loss_act_det.item())
                loss_dict['loss_act_det_train'] = loss_act_det_meter.mean
            # action prediction loss
            if act_pred_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_act_pred = loss_func['act_pred'](act_pred_scores, torch.cat((target_action, target_future_action), dim=1))
                elif cfg.STYLE == 'SF-GRU':
                    loss_act_pred = cross_entropy_loss(act_pred_scores[:, -1].reshape(-1, act_pred_scores.shape[-1]), 
                                                       target_future_action.view(-1))
                loss += loss_act_pred
                loss_act_pred_meter.add(loss_act_pred.item())
                loss_dict['loss_act_pred_train'] = loss_act_pred_meter.mean
            # intent loss
            if int_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_intent = loss_func['int_det'](int_det_scores, target_intent)
                elif cfg.STYLE == 'SF-GRU':
                    loss_intent = loss_func['int_det'](int_det_scores[:, -1:], target_intent[:, -1:])
                if act_det_scores is not None and hasattr(model, 'param_scheduler'):
                    loss += model.param_scheduler.intent_weight * loss_intent
                    loss_dict['intent_weight'] = model.param_scheduler.intent_weight.item()
                else:
                    loss += loss_intent
                loss_intent_meter.add(loss_intent.item())
                loss_dict['loss_int_det_train'] = loss_intent_meter.mean
            # weight
            if hasattr(model, 'param_scheduler'):
                model.param_scheduler.step()

            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # gradient clip
            loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            batch_time = time.time() - end
            loss_dict['batch_time'] = batch_time
            loss_dict['data_time'] = data_time

            # model.param_scheduler.step()
            if cfg.SOLVER.SCHEDULER == 'exp':
                lr_scheduler.step()
            # print log
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(iters, model, loss_dict, optimizer=optimizer, logger=logger, iteration_based=True)
            # visualize
            if cfg.VISUALIZE and iters % 50 == 0 and hasattr(logger, 'log_image'):
                bboxes = bboxes.detach().cpu().numpy()
                if cfg.DATASET.BBOX_NORMALIZE:
                    # NOTE: denormalize bboxes
                    _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
                    _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
                    bboxes = bboxes * (_max - _min) + _min
                
                id_to_show = np.random.randint(bboxes.shape[0])
                gt_behaviors, pred_behaviors = {}, {}
                if 'action' in cfg.MODEL.TASK:
                    target_action = target_action.detach().cpu().numpy()
                    if act_det_scores.shape[-1] == 1:
                        act_det_scores = act_det_scores.sigmoid().detach().cpu().numpy()
                    else:
                        act_det_scores = act_det_scores.softmax(dim=-1).detach().cpu().numpy()
                    gt_behaviors['action'] = target_action[id_to_show, -1]
                    pred_behaviors['action'] = act_det_scores[id_to_show, -1]

                if 'intent' in cfg.MODEL.TASK:
                    target_intent = target_intent.detach().cpu().numpy()
                    if int_det_scores.shape[-1] == 1:
                        int_det_scores = int_det_scores.sigmoid().detach().cpu().numpy()
                    else:
                        int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy() 
                    gt_behaviors['intent'] = target_intent[id_to_show, -1]
                    pred_behaviors['intent'] = int_det_scores[id_to_show, -1]

                # visualize result  
                vis_results(viz, 
                            img_path[id_to_show], 
                            bboxes[id_to_show][-1], 
                            gt_behaviors=gt_behaviors,
                            pred_behaviors=pred_behaviors,
                            name='intent_train',
                            logger=logger)
                
            end = time.time()
            # do validation
            if iters % 100 == 0:
                loss_val = do_val(cfg, iters, model, val_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if cfg.SOLVER.SCHEDULER == 'plateau':
                    lr_scheduler.step(loss_val)
            # do test
            if iters % 250 == 0:
                result_dict = inference(cfg, iters, model, test_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if 'intent' in cfg.MODEL.TASK:
                    save_file = os.path.join(save_checkpoint_dir, 
                                        'iters_{}_acc_{:.3}_f1_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                            result_dict['intent_accuracy'],
                                                                            result_dict['intent_f1']))
                else:
                    save_file = os.path.join(save_checkpoint_dir, 
                                    'iters_{}_mAP_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                        result_dict['mAP']))
                torch.save(model.state_dict(), save_file)
                
                
def do_train_iteration_intent(cfg, model, optimizer, # scaler,
                       train_dataloader, val_dataloader, test_dataloader, 
                       device, logger=None, lr_scheduler=None, save_checkpoint_dir=None):
    model.train()
    max_iters = len(train_dataloader)
    viz = Visualizer(cfg, mode='image')
    # trainning loss meters
    loss_act_det_meter = AverageValueMeter()
    loss_intent_meter = AverageValueMeter()
    # loss functions
    loss_func = {}
    loss_func['act_det'] = binary_cross_entropy_loss if cfg.MODEL.ACTION_LOSS == 'bce' else cross_entropy_loss
    loss_func['int_det'] = binary_cross_entropy_loss if cfg.MODEL.INTENT_LOSS == 'bce' else cross_entropy_loss
    with torch.set_grad_enabled(True):
        end = time.time()
        for iters, batch in enumerate(tqdm(train_dataloader), start=1):
            # result_dict = inference_intent(cfg, iters, model, test_dataloader, device, logger=logger, iteration_based=True) # 为了测试 
            data_time = time.time() - end
            imgs = batch['crop_imgs'].to(device)
            # pedes_imgs = batch['pedes_imgs'].to(device)
            target_intent = batch['obs_intent'].to(device)
            target_action = batch['obs_action'].to(device)
            pose = batch['pose'].to(device)
            # img_path = batch['cur_image_file']
            
            # img_full = batch['full_imgs'].to(device) # 全景图片
            
            # # 只使用crop_imgs            
            # int_det_scores, act_det_scores = model(imgs, False)
            
            # withtraffic
            bboxes = batch['obs_bboxes'].to(device)
            ego_motion = batch['obs_ego_motion'].to(device) if cfg.MODEL.WITH_EGO or cfg.MODEL.WITH_TRAFFIC else None
            x_traffic = None
            
            if cfg.MODEL.WITH_TRAFFIC:
                if cfg.MODEL.PRETRAINED:
                    x_traffic = batch['traffic_features'].to(device)
                else:
                    x_traffic = {}
                    if 'x_neighbor' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_neighbor'] = batch['neighbor_bboxes']
                        x_traffic['cls_neighbor'] = batch['neighbor_classes']

                        
                    if 'x_light' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_light'] = batch['traffic_light']
                        x_traffic['cls_light'] = batch['traffic_light_classes']

                        
                    if 'x_sign' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_sign'] = batch['traffic_sign']
                        x_traffic['cls_sign'] = batch['traffic_sign_classes']

                        
                    if 'x_crosswalk' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_crosswalk'] = batch['crosswalk']
                        x_traffic['cls_crosswalk'] = batch['crosswalk_classes']

                       
                    if 'x_station' in cfg.MODEL.TRAFFIC_TYPES:
                        x_traffic['x_station'] = batch['station']
                        x_traffic['cls_station'] = batch['station_classes']  
                        
                        
            # # 混合精度训练
            # with autocast():
            int_det_scores, act_det_scores = model(x=imgs,
                                                x_pose=pose,
                                                x_bbox=bboxes,
                                                x_ego=ego_motion,
                                                x_traffic=x_traffic)
            
            
            # get loss and update loss meters
            loss, loss_dict = 0.0, {}
            # action detection loss
            if act_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_act_det = loss_func['act_det'](act_det_scores, target_action)
                elif cfg.STYLE == 'SF-GRU':
                    # loss_act_det = loss_func['act_det'](act_det_scores[:, -1:], target_action[:, -1:])
                    loss_act_det = loss_func['act_det'](act_det_scores, target_action[:, -1:])
                loss += loss_act_det
                loss_act_det_meter.add(loss_act_det.item())
                loss_dict['loss_act_det_train'] = loss_act_det_meter.mean
            # intent loss
            if int_det_scores is not None:
                if cfg.STYLE == 'PIE':
                    loss_intent = loss_func['int_det'](int_det_scores, target_intent)
                elif cfg.STYLE == 'SF-GRU':
                    # loss_intent = loss_func['int_det'](int_det_scores[:, -1:], target_intent[:, -1:])
                    loss_intent = loss_func['int_det'](int_det_scores, target_intent[:, -1:])
                if act_det_scores is not None and hasattr(model, 'param_scheduler'):
                    loss += model.param_scheduler.intent_weight * loss_intent
                    loss_dict['intent_weight'] = model.param_scheduler.intent_weight.item()
                else:
                    loss += loss_intent
                loss_intent_meter.add(loss_intent.item())
                loss_dict['loss_int_det_train'] = loss_intent_meter.mean
        
            # weight
            if hasattr(model, 'param_scheduler'):
                model.param_scheduler.step() 
                
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            
            # # # 混合精度
            # scaler.scale(loss).backward()
            loss.backward()
            
            # gradient clip
            loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            
            optimizer.step()
            
            # # 混合精度
            # scaler.step(optimizer)
            # scaler.update()
            
            batch_time = time.time() - end
            loss_dict['batch_time'] = batch_time
            loss_dict['data_time'] = data_time

            # model.param_scheduler.step()
            if cfg.SOLVER.SCHEDULER == 'exp':
                lr_scheduler.step()
                
            # print log
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(iters, model, loss_dict, optimizer=optimizer, logger=logger, iteration_based=True)
            # # visualize
            # if cfg.VISUALIZE and iters % 50 == 0 and hasattr(logger, 'log_image'):
            #     bboxes = bboxes.detach().cpu().numpy()
            #     if cfg.DATASET.BBOX_NORMALIZE:
            #         # NOTE: denormalize bboxes
            #         _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :]
            #         _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
            #         bboxes = bboxes * (_max - _min) + _min
                
            #     id_to_show = np.random.randint(bboxes.shape[0])
            #     gt_behaviors, pred_behaviors = {}, {}
            #     if 'action' in cfg.MODEL.TASK:
            #         target_action = target_action.detach().cpu().numpy()
            #         if act_det_scores.shape[-1] == 1:
            #             act_det_scores = act_det_scores.sigmoid().detach().cpu().numpy()
            #         else:
            #             act_det_scores = act_det_scores.softmax(dim=-1).detach().cpu().numpy()
            #         gt_behaviors['action'] = target_action[id_to_show, -1]
            #         pred_behaviors['action'] = act_det_scores[id_to_show, -1]

            #     if 'intent' in cfg.MODEL.TASK:
            #         target_intent = target_intent.detach().cpu().numpy()
            #         if int_det_scores.shape[-1] == 1:
            #             int_det_scores = int_det_scores.sigmoid().detach().cpu().numpy()
            #         else:
            #             int_det_scores = int_det_scores.softmax(dim=-1).detach().cpu().numpy() 
            #         gt_behaviors['intent'] = target_intent[id_to_show, -1]
            #         pred_behaviors['intent'] = int_det_scores[id_to_show, -1]
                    
            #     # visualize result  
            #     vis_results(viz, 
            #                 img_path[id_to_show], 
            #                 bboxes[id_to_show][-1], 
            #                 gt_behaviors=gt_behaviors,
            #                 pred_behaviors=pred_behaviors,
            #                 name='intent_train',
            #                 logger=logger)    
                
            end = time.time()

            # result_dict = inference_intent(cfg, iters, model, test_dataloader, device, logger=logger, iteration_based=True) # debug
            
            # do validation
            # if iters % 100 == 0:
            if iters % 100 == 0:
                loss_val = do_val(cfg, iters, model, val_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if cfg.SOLVER.SCHEDULER == 'plateau':
                    lr_scheduler.step(loss_val)
            # do test
            # if iters % 100 == 0:
            if iters % 100 == 0:
                result_dict = inference_intent(cfg, iters, model, test_dataloader, device, logger=logger, iteration_based=True)
                model.train()
                if 'intent' in cfg.MODEL.TASK:
                    save_file = os.path.join(save_checkpoint_dir, 
                                        'iters_{}_acc_{:.3}_f1_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                            result_dict['intent_accuracy'],
                                                                            result_dict['intent_f1']))
                else:
                    save_file = os.path.join(save_checkpoint_dir, 
                                    'iters_{}_mAP_{:.3}.pth'.format(str(iters).zfill(3), 
                                                                        result_dict['mAP']))
                torch.save(model.state_dict(), save_file)