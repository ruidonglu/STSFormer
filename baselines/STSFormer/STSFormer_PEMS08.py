import os
import sys
import torch
import numpy as np
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.archs import STSFormer
from basicts.utils import load_adj, get_Laplacian
CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STSFormer model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS08"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STSFormer"
CFG.MODEL.ARCH = STSFormer
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "normlap")
adj_mx = np.array(adj_mx[0])
lap_mx = get_Laplacian(adj_mx)
CFG.MODEL.PARAM = {
    "DEVICE" : 'cuda:0',
    "in_channels" : 8,
    "adj_mx":adj_mx,
    "dropout" : 0,
    "alpha" : 2,
    "lap_mx": lap_mx,
    "kernel_size" : 3,
    "t_attn_size" : 8,
    "t_num_heads" : 8,
    "output_size" : 1,
    "feature_dim" : 1,
    "embed_dim" : 96,
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    "if_T_o_D": True,
    "if_D_o_W": True,
    "input_dim": 3,
    "encoder_num": 4,
    'temp_dim_tod':16,
    'temp_dim_dow':16,
    'if_spatial':True,
    "num_nodes": 170,
    'node_dim':32,
    'series_dim':32
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2] # traffic flow, time in day
CFG.MODEL.TARGET_FEATURES = [0] # traffic flow

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
