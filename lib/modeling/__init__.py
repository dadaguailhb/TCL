from .conv3d_based.act_intent import ActionIntentionDetection as Conv3dModel
from .rnn_based.model import ActionIntentionDetection as RNNModel
from .relation.relation_embedding import RelationEmbeddingNet
from .vit.vit_model import VisionTransformer
from .vit.classifier import Classifier,Classifier_Withtraffic,Classifier_VitPreModel
from .vit.timm_model import *


def make_model(cfg):
    if cfg.MODEL.TYPE == 'conv3d':
        model = Conv3dModel(cfg)
    elif cfg.MODEL.TYPE == 'rnn':
        model = RNNModel(cfg)
    elif cfg.MODEL.TYPE == 'relation':
        model = RelationEmbeddingNet(cfg)
    elif cfg.MODEL.TYPE == 'vit':
        if cfg.finetune or cfg.pretrain:
            model = Classifier_VitPreModel(cfg)
        else:
            model = VisionTransformer(
                img_size=cfg.img_size,
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
    else:
        raise NameError("model type:{} is unknown".format(cfg.MODEL.TYPE)) 

    return model
