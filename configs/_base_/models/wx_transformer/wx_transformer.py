"""PointNet-Transformer model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'wx_transformer'
_C.rot_type = 'quat'
_C.pc_feat_dim = 256

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

_C.transformer_feat_dim = 1024
_C.transformer_heads = 8
_C.transformer_layers = 4
_C.transformer_dropout = 0.1

_C.share_vanilla_parameters = False
_C.shared_memory_attention = False
_C.use_topk = False
_C.topk = 5
_C.mem_slots = 8
_C.num_steps = 100
_C.null_attention = False


def get_cfg_defaults():
    return _C.clone()
