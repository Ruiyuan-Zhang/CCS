import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../../../../_base_/default_exp.py',
    'data': '../../../../_base_/datasets/partnet/partnet_chair.py',
    'optimizer': '../../../../_base_/schedules/adam_cosine.py',
    'model': '../../../../_base_/models/wx_transformer/wx_transformer.py',
    'loss': '../../../../_base_/models/loss/semantic_loss.py',
}

# Miscellaneous configs
_C = CN()

_C.exp = CN()
_C.exp.num_epochs = 400
_C.exp.batch_size = 16 # 32

_C.optimizer = CN()
_C.optimizer.warmup_ratio = 0.05
_C.optimizer.lr = 1e-3

_C.model = CN()
_C.model.share_vanilla_parameters = False
_C.model.shared_memory_attention = False
_C.model.use_topk = True
_C.model.topk = 1
_C.model.mem_slots = 8
_C.model.num_steps = 100
_C.model.null_attention = False

_C.loss = CN()
_C.loss.collision_loss_w = 0.0
_C.loss.collision_loss_C = 0.1

def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg
