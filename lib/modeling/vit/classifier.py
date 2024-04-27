# import sys,os
# sys.path.append(os.pardir)
# sys.path.append("./")

from torch import nn
import torch
import torch.nn.functional as F
from .pretrain_model import Pretrain_model
from typing import Optional
from .config import resolve_config
from lib.modeling.relation import RelationNet
from .positional_embedding import TrafficNet,PoseNet
from timm.models import create_model
import torch
from collections import OrderedDict
import utils
from .timm_model import vit_base_patch16_224

import os
from einops import rearrange
from PIL import Image
import time
import numpy as np

class Classifier(nn.Module):
    
    def __init__(self,
                 model_ckpt: Optional[str] = None
                 ):
        super().__init__()
        
        # self.model = Pretrain_model.from_file('marlin_vit_small', model_ckpt).encoder
        self.model = Pretrain_model.from_file('marlin_vit_base', model_ckpt).encoder
        # self.config = resolve_config('marlin_vit_small')
        self.config = resolve_config('marlin_vit_base')
        self.conv = nn.Conv1d(in_channels=1568, out_channels=16, kernel_size=(1,), stride=(1,), padding=(0,))
        self.classifier1 = nn.Linear(self.config.encoder_embed_dim, 1)
        self.classifier2 = nn.Linear(self.config.encoder_embed_dim, 7) # num_action
    
    def extract_features(self, x, seq_mean_pool=False):
        if self.model is not None:
            feat = self.model.extract_features(x, seq_mean_pool)  # False输出应该是(batch_size, 588, 384) 如果是True，那么就是(batch_size, 384)
        else:
            feat = x
            
        # feat = self.conv(feat)
        
        # feat = feat[:, 195::196, :]
        # feat = torch.repeat_interleave(feat,2,dim=1)
        
        return feat
        
    def forward(self, x, seq_mean_pool=False):
        if self.model is not None:
            feat = self.model.extract_features(x, seq_mean_pool)  # False输出应该是(batch_size, 588, 384) 如果是True，那么就是(batch_size, 384)
        else:
            feat = x
        feat = self.conv(feat)
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)
        return score1, score2
        
class Classifier_full(nn.Module):
    
    def __init__(self,
                 model_ckpt: Optional[str] = None
                 ):
        super().__init__()
        
        self.model = Pretrain_model.from_file('marlin_vit_small', model_ckpt).encoder
        self.config = resolve_config('marlin_vit_small')
        self.conv = nn.Conv1d(in_channels=1568, out_channels=16, kernel_size=(1,), stride=(1,), padding=(0,))
        self.classifier1 = nn.Linear(self.config.encoder_embed_dim, 1)
        self.classifier2 = nn.Linear(self.config.encoder_embed_dim, 7) # num_action
    
    def extract_features(self, x, seq_mean_pool):
        if self.model is not None:
            feat = self.model.extract_features(x, seq_mean_pool)  # False输出应该是(batch_size, 588, 384) 如果是True，那么就是(batch_size, 384)
        else:
            feat = x
        feat = self.conv(feat)
        return feat
        
    def forward(self, x, seq_mean_pool):
        if self.model is not None:
            feat = self.model.extract_features(x, seq_mean_pool)  # False输出应该是(batch_size, 588, 384) 如果是True，那么就是(batch_size, 384)
        else:
            feat = x
        feat = self.conv(feat)
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)
        return score1, score2
    
class Classifier_Withtraffic(nn.Module):
    
    def __init__(self,
                 cfg
                 ):
        super().__init__()
        self.cfg = cfg
        self.model_ckpt = cfg.model_ckpt
        self.img_model = Classifier(self.model_ckpt)
        # self.img_full_model = Classifier_full(self.model_ckpt)
        self.relation_model = RelationNet(self.cfg)
        self.traffic_model = TrafficNet(
            data_type=cfg.DATASET.NAME,
            patch_size=cfg.patch_size,
            n_frames=cfg.clip_frames,
            embed_dim=cfg.encoder.embed_dim,
            depth=cfg.encoder.depth,
            num_heads=cfg.encoder.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            tubelet_size=cfg.tubelet_size
        )
        self.weight1 = nn.Parameter(torch.ones(2)) # 使用全景图片，由2改为3
        
        # self.conv = nn.Conv1d(in_channels=1568, out_channels=16, kernel_size=(1,), stride=(1,), padding=(0,))
        self.classifier1 = nn.Linear(self.cfg.encoder.embed_dim, 1)
        self.classifier2 = nn.Linear(self.cfg.encoder.embed_dim, 7) # num_action
     
    def embed_traffic_features(self, x_bbox, x_ego, x_traffic):
        x_traffic['x_ego'] = x_ego
        if self.cfg.DATASET.NAME == 'PIE':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        elif self.cfg.DATASET.NAME == 'JAAD':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        else:
            raise NameError(self.cfg.DATASET.NAME)
    
    def concat_traffic_features(self):
        return self.relation_model.concat_traffic_features()
    
     
    def forward(self,
                x,
                x_full=None, # 全景
                x_bbox=None,
                x_ego=None,
                x_traffic=None,
                seq_mean_pool=False):
        # x_full = self.img_full_model.extract_features(x_full, seq_mean_pool)
        # x_full = self.img_model.extract_features(x_full, seq_mean_pool)
        x = self.img_model.extract_features(x, seq_mean_pool)
        
        self.embed_traffic_features(x_bbox, x_ego, x_traffic)
        x_traffic = self.concat_traffic_features() # 得到(b, 16, 224 or 192)
        x_traffic = self.traffic_model(x_traffic)
        
        # 加权两个网络输出
        weight_normalized1 = F.softmax(self.weight1, dim=0)
        feat = weight_normalized1[0] * x + weight_normalized1[1] * x_traffic # + weight_normalized1[2] * x_full
        
        # feat = x + x_traffic
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)
        return score1, score2
    # def forward(self, x, seq_mean_pool=False):
    #     if self.model is not None:
    #         feat = self.model.extract_features(x, seq_mean_pool)  # False输出应该是(batch_size, 588, 384) 如果是True，那么就是(batch_size, 384)
    #     else:
    #         feat = x
    #     feat = self.conv(feat)
    #     score1 = self.classifier1(feat)
    #     score2 = self.classifier2(feat)
    #     return score1, score2
  
  
class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        
        # Linear layer
        self.linear = nn.Linear(in_features=196, out_features=1)
        
    def forward(self, x):
        # Permute dimensions
        x = x.reshape(-1,8,196,768)
        x = x.permute(0, 3, 1, 2)  # (batch_size, 8, 768, 196)
        
        
        # Linear layer
        x = self.linear(x)
        
        
        # Remove the last dimension
        x = x.squeeze(-1)
        
        # Permute back
        x = x.permute(0, 2, 1)  # (batch_size, 8, 768)
        x = torch.repeat_interleave(x,2,dim=1)
        
        return x  

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
 
def cluster_dpc_knn(token, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        # x = token_dict["x"]
        x = token
        B, N, C = x.shape

        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)  # (B, N, N)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        # k最小是2，因为k=1时，每个点最近的一个点只有他自己，没办法计算密度
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False) # 返回的 形状是(B,N,K)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # ->(B,N)
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]  # （B,1,1）
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density # ->(B,N)
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1) # eg. [[2,2,1,0,0,0,0,0],[...]]

    return idx_cluster
 
def frame_merge(token, idx_cluster, cluster_num):
    B, N, C = token.shape
    group_data = []
    for i in range(B):
        group_batch = []
        token_batch = token[i]
        for j in range(cluster_num):
            temp = token_batch[idx_cluster[i] == j]
            group_batch.append(temp.mean(dim=0))
        group_batch = torch.stack(group_batch, dim=0)
        group_data.append(group_batch)
    group_data = torch.stack(group_data, dim=0) # (B,cluster_num,768)
    
    return group_data
    
    
class Temporal_Merging(nn.Module):
    def __init__(self, n, k):
        super(Temporal_Merging, self).__init__()
        self.cluster_num = n # 3
        self.k = k # 5
        # self.merging = nn.Linear(3, 1)
        
    def forward(self, x):
        # x = x.reshape(-1, 8, 196, 768)
        # x = x.mean(dim=2)
        # x = x.squeeze(2)
        idx_cluster = cluster_dpc_knn(token=x, cluster_num=self.cluster_num, k=self.k)
        feat = frame_merge(token=x, idx_cluster=idx_cluster, cluster_num=self.cluster_num)
        # feat = self.merging(feat.permute(0,2,1))
        # feat = feat.sequeeze(-1)
        return feat
        
        
    
class Classifier_VitPreModel(nn.Module): # pretrained vit_base model
    
    def __init__(self,
                 cfg
                 ):
        super().__init__()
        self.cfg = cfg
        
        # mars and muke pretrain
        # self.model_ckpt = cfg.model_ckpt
        # self.img_model = Classifier(self.model_ckpt)
        # self.pedes_model = Classifier(self.model_ckpt)
        # self.img_full_model = Classifier_full(self.model_ckpt)
        
        # k400 pretrain
        model = vit_base_patch16_224()
        if cfg.finetune:
            if cfg.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    cfg.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(cfg.finetune, map_location='cpu')

            print("Load ckpt from %s" % cfg.finetune)
            checkpoint_model = None
            for model_key in cfg.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
                num_patches = model.patch_embed.num_patches # 
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

                # height (== width) for the checkpoint position embedding 
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(cfg.clip_frames // model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int((num_patches // (cfg.clip_frames // model.patch_embed.tubelet_size) )** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(-1, cfg.clip_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, cfg.clip_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                    pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed

            utils.load_state_dict(model, checkpoint_model, prefix=cfg.model_prefix)
            print('successfully loaded vit_base pretrained model!')        
        self.img_model = model
        
        self.relation_model = RelationNet(self.cfg)
        self.traffic_model = TrafficNet(
            data_type=cfg.DATASET.NAME,
            patch_size=cfg.patch_size,
            n_frames=cfg.clip_frames,
            embed_dim=cfg.encoder.embed_dim,
            depth=cfg.encoder.depth,
            num_heads=cfg.encoder.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            tubelet_size=cfg.tubelet_size
        )
        # self.pose_model = PoseNet(
        #     data_type=cfg.DATASET.NAME,
        #     patch_size=cfg.patch_size,
        #     n_frames=cfg.clip_frames,
        #     embed_dim=cfg.encoder.embed_dim,
        #     depth=cfg.encoder.depth,
        #     num_heads=cfg.encoder.num_heads,
        #     mlp_ratio=cfg.mlp_ratio,
        #     qkv_bias=cfg.qkv_bias,
        #     qk_scale=cfg.qk_scale,
        #     tubelet_size=cfg.tubelet_size
        # )
        # self.weight1 = nn.Parameter(torch.ones(2)) # 使用了全景图片，由2改为3
        
        # self.concat = CustomNetwork()
        self.conv = nn.Conv1d(in_channels=1568, out_channels=16, kernel_size=(1,), stride=(1,), padding=(0,))
        self.tem_merge_x = Temporal_Merging(3, 5)
        self.tem_merge_traffic = Temporal_Merging(3, 8)
        self.linear = nn.Linear(768, 384)
        
        # using attention block
        self.attention_x = Attention3DBlock(self.cfg.encoder.embed_dim)
        self.attention_traffic = Attention3DBlock(self.cfg.encoder.embed_dim)
        self.attention_fusion = Attention3DBlock(128)
        self.classifier1 = nn.Linear(128, 1)
        # self.classifier2 = nn.Linear(128, 7)
        
        # self.classifier1 = nn.Linear(self.cfg.encoder.embed_dim, 1)
        # self.classifier2 = nn.Linear(self.cfg.encoder.embed_dim, 7) # num_action
     
    def embed_traffic_features(self, x_bbox, x_ego, x_traffic):
        x_traffic['x_ego'] = x_ego
        if self.cfg.DATASET.NAME == 'PIE':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        elif self.cfg.DATASET.NAME == 'JAAD':
            self.relation_model.embed_traffic_features(x_bbox, x_traffic)
        else:
            raise NameError(self.cfg.DATASET.NAME)
    
    def concat_traffic_features(self):
        return self.relation_model.concat_traffic_features()
    
     
    def forward(self,
                x,
                x_pose=None,
                x_pedes=None,
                x_full=None, # 全景
                x_bbox=None,
                x_ego=None,
                x_traffic=None,
                seq_mean_pool=False):
        # x_full = self.img_full_model.extract_features(x_full, seq_mean_pool)       
        # x = self.img_model.extract_features(x)
        
        # # 创建保存路径
        # save_dir = "./img_crop_image"       
        # os.makedirs(save_dir, exist_ok=True)
        # frame = rearrange(x, "b c t h w ->b t c h w")
        # frame = frame.cpu()
        # for i in range(frame.shape[0]):
        #     for j in range(frame.shape[1]):        
        #         # 保存裁剪后的图片到本地
        #         img = frame[i][j]
        #         img = (img.numpy()*255).astype(np.uint8)
        #         img = np.transpose(img, (1,2,0))
        #         img = Image.fromarray(img)
        #         timestamp = int(time.time()*1000)
        #         save_path_org = os.path.join(save_dir, f'{timestamp}.png')
        #         img.save(save_path_org)
        #         print('保存成功')
        
        # with torch.no_grad():
        # k400 pretrian
        x = self.img_model.forward_features(x)
        
        # using temporal merging
        x = x.reshape(-1,8,196,768)
        x = x.mean(dim=2)
        x = x.squeeze(2)
        # 消融
        # x = self.tem_merge_x(x)
        
        # x = self.conv(x)
        # x = x[:, ::196, :]
        # x = torch.repeat_interleave(x,2,dim=1)
        
        x = self.linear(x)
  
        # using attention block
        x = self.attention_x(x)
        
        
        # # 使用pedes（无context）
        # x_pedes = self.pedes_model.extract_features(x_pedes)
        # x_pedes = x_pedes[:, ::196, :]
        # x_pedes = torch.repeat_interleave(x_pedes,2,dim=1)
        
        # print('------using conv-------')
        # x = self.conv(x)
        # x = self.concat(x) # linear
        
        self.embed_traffic_features(x_bbox, x_ego, x_traffic)
        x_traffic = self.concat_traffic_features() # 得到(b, 16, 224 or 192)
        # x_traffic = self.traffic_model(x_traffic)
        x_traffic = self.traffic_model(x_traffic, x_pose) 
        
        # using event merging
        # x_traffic = self.tem_merge_traffic(x_traffic)
        
        # using attention block
        x_traffic = self.attention_traffic(x_traffic)
        
        # x_traffic = self.traffic_model(x_traffic)# fusion attention
        # x_pose = self.pose_model(x_pose)   # plus attention
        
        # # 加权两个网络输出
        # weight_normalized1 = F.softmax(self.weight1, dim=0)
        # feat = weight_normalized1[0] * x + weight_normalized1[1] * (x_traffic + x_pose) 
        
        # feat = x + x_pedes + x_traffic  # 
        # feat = x + x_traffic # + x_pose  # 可以沿着最后一个维度进行cat，然后通过一个线性层映射到384，后面可以尝试
        
        # using attention block
        feat = torch.stack((x,  x_traffic), dim=1)
        feat = self.attention_fusion(feat)
        
        score1 = self.classifier1(feat)
        # score2 = self.classifier2(feat)
        return score1, None
    
class Attention3DBlock(nn.Module):
    def __init__(self, input_dim, dense_size=128):
        super(Attention3DBlock, self).__init__()
        self.input_dim = input_dim
        self.dense_size = dense_size
        
        self.score_first_part = nn.Linear(input_dim, input_dim, bias=False)
        self.last_hidden_state = nn.Linear(input_dim, 1, bias=False)
        self.context_vector = nn.Linear(input_dim * 2, dense_size, bias=False)
        
    def forward(self, hidden_states):
        batch_size, time_steps, _ = hidden_states.size()
        
        # Calculate attention score
        score_first_part = self.score_first_part(hidden_states)
        h_t = hidden_states[:, -1, :]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # -> (batch_size, t)
        attention_weights = F.softmax(score, dim=1)
        
        # Calculate context vector (b,t,c) (b,t)
        context_vector = torch.bmm(hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2) #  -> (b,c)
        
        # Concatenate context vector and last hidden state
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        
        # Final attention vector
        attention_vector = torch.tanh(self.context_vector(pre_activation)) # -> (b, 128)
        
        return attention_vector