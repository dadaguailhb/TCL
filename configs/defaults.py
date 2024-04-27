import os

from yacs.config import CfgNode as CN

_C = CN()

_C.USE_WANDB = False
_C.PROJECT = 'intent2021icra'
_C.CKPT_DIR = 'checkpoints/PIE'
_C.OUT_DIR = 'outputs/PIE'
_C.DEVICE = 'cuda'
_C.GPU = '0'
_C.VISUALIZE = False
_C.PRINT_INTERVAL = 10
_C.STYLE = 'PIE'

# ------ MODEL ---
_C.MODEL = CN()
_C.MODEL.TYPE = 'rnn'
_C.MODEL.TASK = 'action_intent'
_C.MODEL.PRETRAINED = False # whether to use pre-trained relation embedding or not.
# _C.MODEL.INTENT_ONLY = True
_C.MODEL.WITH_EGO = False
_C.MODEL.WITH_TRAFFIC = False
_C.MODEL.WITH_POSE = False
_C.MODEL.TRAFFIC_ATTENTION = 'none'
_C.MODEL.TRAFFIC_TYPES = []
_C.MODEL.INPUT_LAYER = 'avg_pool'
_C.MODEL.ACTION_NET = 'gru'
_C.MODEL.ACTION_NET_INPUT = 'pooled'
_C.MODEL.ACTION_LOSS = 'ce'
_C.MODEL.INTENT_NET = 'gru'
_C.MODEL.INTENT_LOSS = 'bce'
_C.MODEL.CONVLSTM_HIDDEN = 64

_C.MODEL.SEG_LEN = 30
_C.MODEL.INPUT_LEN = 15
_C.MODEL.PRED_LEN = 5
_C.MODEL.HIDDEN_SIZE = 128
_C.MODEL.DROPOUT = 0.4
_C.MODEL.RECURRENT_DROPOUT = 0.2
_C.MODEL.ROI_SIZE = 7
_C.MODEL.POOLER_SCALES = (0.03125,)
_C.MODEL.POOLER_SAMPLING_RATIO = 0

# ------ DATASET -----
_C.DATASET = CN()
_C.DATASET.NAME = 'PIE'
_C.DATASET.ROOT = 'data/PIE_dataset'
_C.DATASET.FPS = 30
_C.DATASET.NUM_ACTION = 2
_C.DATASET.NUM_INTENT = 2
_C.DATASET.BALANCE = False
_C.DATASET.MIN_BBOX = [0,0,0,0] # the min of cxcywh or x1x2y1y2
_C.DATASET.MAX_BBOX = [1920, 1080, 1920, 1080] # the max of cxcywh or x1x2y1y2
_C.DATASET.FPS = 30
_C.DATASET.OVERLAP = 0.5
_C.DATASET.BBOX_NORMALIZE = False
# ------ SOLVER ------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 10
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.MAX_ITERS = 10000
_C.SOLVER.LR = 1e-5
_C.SOLVER.SCHEDULER = ''
_C.SOLVER.GAMMA = 0.9999
_C.SOLVER.L2_WEIGHT = 0.001
_C.SOLVER.INTENT_WEIGHT_MAX = -1
_C.SOLVER.CENTER_STEP = 500.0
_C.SOLVER.STEPS_LO_TO_HI = 100.0
# ----- TEST ------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.INTERVAL = 5

# ------ DATALOADER ------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 1
_C.DATALOADER.ITERATION_BASED = False
_C.DATALOADER.WEIGHTED = 'none'

# ---vit----
_C.pretrain = False
_C.model_ckpt = ''
_C.finetune_model = 'vit_base_patch16_224'
_C.finetune = None
_C.model_key = 'model|module'
_C.model_prefix = ''

_C.img_size= 224
_C.patch_size= 16
_C.clip_frames= 16
_C.tubelet_size= 2
_C.mask_strategy= "tube"
_C.temporal_sample_rate= 2
_C.mask_percentage_target= 0.9
_C.mlp_ratio= 4.0
_C.qkv_bias= True
_C.qk_scale= None
_C.drop_rate= 0.0
_C.attn_drop_rate= 0.0
_C.norm_layer= "LayerNorm"
_C.init_values= 0.0
_C.weight_decay= 0.0
_C.feature_dir= "Marlin_Features_Vit_Small"
_C.adv_loss= True
_C.adv_weight= 0.01
_C.gp_weight= 0.0
_C.d_steps= 1
_C.g_steps= 1

_C.learning_rate= CN()
_C.learning_rate.base= 1.5e-4
_C.learning_rate.warmup= 1.0e-6
_C.learning_rate.min= 1.0e-5
_C.learning_rate.warmup_epochs=40

_C.optimizer= CN()
_C.optimizer.type= "AdamW"
_C.optimizer.eps= 1.0e-8
_C.optimizer.betas= [0.9, 0.95]

_C.encoder= CN()
_C.encoder.embed_dim= 384
_C.encoder.depth= 12
_C.encoder.num_heads= 6