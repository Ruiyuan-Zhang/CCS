"""PointNet-Transformer model with iterative refinement."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'wx_transformer_refine'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

# transformer basic configuration
_C.transformer_pos_enc = (128, 128)
_C.transformer_feat_dim = 512
_C.transformer_heads = 8
_C.transformer_layers = 2
_C.transformer_dropout = 0.1

# workspace
_C.share_vanilla_parameters = False
_C.shared_memory_attention = False
_C.use_topk = False
_C.topk = 5
_C.mem_slots = 8
_C.num_steps = 100
_C.null_attention = False

# refine
_C.pose_pc_feat = True  # pose regressor input part points feature
_C.refine_steps = 3

def get_cfg_defaults():
    return _C.clone()
