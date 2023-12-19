import torch

from .transforms import rot_pc, transform_pc
from .chamfer import chamfer_distance


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_l2_loss(trans1, trans2, valids):
    """L2 loss for transformation.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_l2_loss(rot1, rot2, valids):
    """L2 loss for rotation.

    Args:
        rot1: [B, P, 4], Rotation3D, should be quat
        rot2: [B, P, 4], Rotation3D, should be quat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    assert rot1.rot_type == rot2.rot_type == 'quat'
    quat1, quat2 = rot1.rot, rot2.rot
    # since quat == -quat
    rot_l2_1 = (quat1 - quat2).pow(2).sum(dim=-1)  # [B, P]
    rot_l2_2 = (quat1 + quat2).pow(2).sum(dim=-1)
    loss_per_data = torch.minimum(rot_l2_1, rot_l2_2)
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_cosine_loss(rot1, rot2, valids):
    """Cosine loss for rotation.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    assert rot1.rot_type == rot2.rot_type
    rot_type = rot1.rot_type
    # cosine distance
    if rot_type == 'quat':
        quat1, quat2 = rot1.rot, rot2.rot
        loss_per_data = 1. - torch.abs(torch.sum(quat1 * quat2, dim=-1))
    # |I - R1^T @ R2|^2
    elif rot_type == 'rmat':
        B = rot1.shape[0]
        rmat1, rmat2 = rot1.rot.view(-1, 3, 3), rot2.rot.view(-1, 3, 3)
        iden = torch.eye(3).unsqueeze(0).type_as(rmat1)
        loss_per_data = (iden - torch.bmm(rmat1.transpose(1, 2), rmat2)).\
            pow(2).mean(dim=[-1, -2]).view(B, -1)
    else:
        raise NotImplementedError(f'cosine loss not supported for {rot_type}')
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_points_l2_loss(pts, rot1, rot2, valids, ret_pts=False):
    """L2 distance between point clouds transformed by rotations.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    pts1 = rot_pc(rot1, pts)
    pts2 = rot_pc(rot2, pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def rot_points_cd_loss(pts, rot1, rot2, valids, ret_pts=False):
    """Chamfer distance between point clouds transformed by rotations.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the rotated point clouds

    Returns:
        [B], loss per data in the batch
    """
    B = pts.shape[0]

    pts1 = rot_pc(rot1, pts)
    pts2 = rot_pc(rot2, pts)

    dist1, dist2 = chamfer_distance(pts1.flatten(0, 1), pts2.flatten(0, 1))
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(B, -1).type_as(pts)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data

def collision_loss(
    pts, 
    trans1,
    rot1,
    valids,
    C = 1.0,             # 用以描述弹性方程，两个云点之间的距离越近，loss越大
    epsilon = 1e-5,      # 一个极小值，用来解决log(0)的问题
):
    pts1 = transform_pc(trans1, rot1, pts)

    center_pts = pts1.sum(dim=2)/pts1.shape[2]
    distances = torch.cdist(center_pts, center_pts, p=2)                              # 计算这些中心点之间的距离 distance.shape = [32,20,20]

    collision_loss_version = 2
    if collision_loss_version == 1:
        distances = 1 - torch.log(C * distances + epsilon)
    elif collision_loss_version == 2:
        distances = torch.exp(2 - C * distances)


    
    # distances = torch.clamp(distances, min=0)    如何采用这个min=0来进行截断，其实会影响其性能，我觉得这是因为下面这部分的均值的计算，使得model让所有的part之间都远离了。
    
    # 对没有用到的零件进行掩码
    mask_distcances = torch.where(valids==1, False, True).unsqueeze(-1)               # 构造一个mask向量，有一些零件是不参与计算的
    masked_distances = torch.masked_fill(distances, mask_distcances, 0)
    masked_distances = masked_distances.permute(0, 2, 1)
    masked_distances = torch.masked_fill(masked_distances, mask_distcances, 0)
    masked_distances = masked_distances.permute(0, 2, 1)

    # 对掩码之后的结果求均值作为loss
    masked_distances = torch.tril(masked_distances, diagonal=-1)                      # 掩盖上三角及对角线上的值
    distances_sum = masked_distances.reshape(mask_distcances.shape[0],-1).sum(dim=1)
    dist = distances_sum/(valids.sum(dim=1)*(valids.sum(dim=1)-1))                    # 计算均值

    # TODO: 是不是还可以尝试使用max结果作为loss

    return dist

def shape_cd_loss(
    pts,
    trans1,
    trans2,
    rot1,
    rot2,
    valids,
    ret_pts=False,
    training=True,
):
    """Chamfer distance between point clouds after rotation and translation.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        ret_pts: whether to return the transformed point clouds
        training: at training time we divide the SCD by `P` as an automatic
            hard negative mining strategy; while at test time we divide by
            the correct number of parts per shape

    Returns:
        [B], loss per data in the batch
    """
    B, P, N, _ = pts.shape

    # fill the padded points with very large numbers
    # so that they won't be matched to any point in CD
    # clone the points to avoid changing the original points
    pts = pts.detach().clone()
    valid_mask = valids[..., None, None]  # [B, P, 1, 1]
    pts = pts.masked_fill(valid_mask == 0, 1e3)

    pts1 = transform_pc(trans1, rot1, pts)
    pts2 = transform_pc(trans2, rot2, pts)

    shape1 = pts1.flatten(1, 2)  # [B, P*N, 3]
    shape2 = pts2.flatten(1, 2)
    dist1, dist2 = chamfer_distance(shape1, shape2)  # [B, P*N]

    
    # pts1 表示云点图，类型是 torch.Tensor, shape = [32, 20, 1000, 3], 分别表示[batch_size, 零件的最大数量, 云点数量，坐标]
    # 计算每个零件的中心点
    # 计算任意两个零件之间中心点的欧式距离 math.sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
    # 求得距离之和，并应用 loss = e^(1-x/c), c=1 来计算loss

    valids = valids.float().detach()
    if training:
        valids = valids.unsqueeze(2).repeat(1, 1, N).view(B, -1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        # we divide the loss by a fixed number `P`
        # this is actually an automatic hard negative loss weighting mechanism
        # shapes with more parts will have higher loss
        # ablation shows better results than using the correct SCD for training
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    # this is the correct SCD calculation
    else:
        valids = valids.float().detach()
        dist = (dist1 + dist2).view(B, P, N).mean(-1)  # [B, P]
        loss_per_data = _valid_mean(dist, valids)

    if ret_pts:
        return loss_per_data, pts1, pts2
    return loss_per_data


def repulsion_cd_loss(part_pcs, valids, thre):
    """Compute Chamfer distance as repulsion loss to push parts away.

    Args:
        part_pcs: [B, P, N, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        thre: float, only penalize cd smaller than this value

    Returns:
        [B], loss per data in the batch
    """
    B, P, N, _ = part_pcs.shape
    pts1 = part_pcs.unsqueeze(2).repeat_interleave(P, dim=2).flatten(0, 2)
    pts2 = part_pcs.unsqueeze(1).repeat_interleave(P, dim=1).flatten(0, 2)
    dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P*P, N]
    cd = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)  # [B*P*P]
    cd = torch.clamp_min(thre - cd.view(B, P, P), min=0.)  # [B, P, P]
    valid_mask = torch.ones(B, P, P).type_as(valids) * \
        valids[:, :, None] * valids[:, None, :]
    loss_per_data = (cd * valid_mask).sum([1, 2]) / valid_mask.sum([1, 2])
    return loss_per_data
