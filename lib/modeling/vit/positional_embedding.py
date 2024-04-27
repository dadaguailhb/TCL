import torch
from torch import Tensor, nn
from torch.nn import ModuleList, LayerNorm
from .modules import Block
from .modules import Shape


class TrafficNet(nn.Module):
    def __init__(self, data_type, patch_size=16, n_frames=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2):
        super(TrafficNet, self).__init__()
        if data_type == 'PIE':
            self.linear = nn.Linear(224+36, embed_dim)  # 可学习的线性层，224为7个32维度的traffic特征合并得到的,36是pose
        else:
            self.linear = nn.Linear(192+36, embed_dim)
            
        
        self.layer_norm = nn.LayerNorm(embed_dim)  # 层归一化
        
        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")
        
        # 位置编码
        self.pos_embedding = SinCosPositionalEmbedding((patch_size, embed_dim), dropout_rate=0.)
        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])   
        
        # # for mlp
        # self.traffic_model_mlp = nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, embed_dim)
        # )    

    def forward(self, feat, pose=None):
        if pose != None:
            feat = torch.cat((feat, pose), dim=-1)
        # 应用线性变换
        feat = self.linear(feat) # (b, 16, 384)
        # 添加位置编码
        feat = self.pos_embedding(feat)
        for block in self.blocks:
            feat = block(feat)
        feat = self.norm(feat)       
        
        # x = self.traffic_model_mlp(x)
        # x = self.norm(x) 
        return feat

class PoseNet(nn.Module):
    def __init__(self, data_type, patch_size=16, n_frames=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2):
        super(PoseNet, self).__init__()
          
        self.pose_embedding = nn.Linear(36, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)  # 层归一化
        
        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")
        
        # 位置编码
        self.pos_embedding = SinCosPositionalEmbedding((patch_size, embed_dim), dropout_rate=0.)
        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])   
        
        # # for mlp
        # self.traffic_model_mlp = nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, embed_dim)
        # )    

    def forward(self, feat):
        # feat = torch.cat((x, pose), dim=-1)
        # 应用线性变换
        feat = self.pose_embedding(feat) # (b, 16, 384)
        # 添加位置编码
        feat = self.pos_embedding(feat)
        for block in self.blocks:
            feat = block(feat)
        feat = self.norm(feat)       
        
        # x = self.traffic_model_mlp(x)
        # x = self.norm(x) 
        return feat

class PositionalEmbedding(nn.Module):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class SinCosPositionalEmbedding(PositionalEmbedding):

    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(
                2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()
