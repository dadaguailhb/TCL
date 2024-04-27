import torch
from torch import nn, Tensor
from torch.nn import Linear
from typing import Generator, Optional

from .decoder import Decoder
from .encoder import Encoder

from .config import resolve_config

class Pretrain_model(nn.Module):

    def __init__(self,
        img_size: list,
        patch_size: int,
        n_frames: int,
        encoder_embed_dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop_rate: float,
        attn_drop_rate: float,
        norm_layer: str,
        init_values: float,
        tubelet_size: int,
        as_feature_extractor: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.as_feature_extractor = as_feature_extractor
        self.clip_frames = n_frames
        if as_feature_extractor:
            self.enc_dec_proj = None
            self.decoder = None
        else:
            self.decoder = Decoder(
                img_size=img_size,
                patch_size=patch_size,
                n_frames=n_frames,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size
            )
            
        self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self.as_feature_extractor:
            raise RuntimeError("For feature extraction, please use `extract_features` or `extract_video`.")
        else:
            assert mask is not None
            x = self.encoder(x, mask)
            x = self.enc_dec_proj(x)
            x = self.decoder(x, mask)
        return x
    
    @classmethod
    def from_file(cls, model_name: str, path: str) -> "Pretrain_model":
        if path.endswith(".pt"):
            state_dict = torch.load(path, map_location="cpu")
        elif path.endswith(".ckpt"):
            state_dict = torch.load(path, map_location="cpu")["state_dict"]

            discriminator_keys = [k for k in state_dict.keys() if k.startswith("discriminator")]
            for key in discriminator_keys:
                del state_dict[key]
        else:
            raise ValueError(f"Unsupported file type: {path.split('.')[-1]}")
        # determine if the checkpoint is full model or encoder only.
        for key in state_dict.keys():
            if key.startswith("decoder."):
                as_feature_extractor = False
                break
        else:
            as_feature_extractor = True

        config = resolve_config(model_name)
        model = cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=config.n_frames,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
            as_feature_extractor=as_feature_extractor
        )
        model.load_state_dict(state_dict)
        print('successfully load pretrain model!')
        return model