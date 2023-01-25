from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelResBackBone8x_multiframe
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_focal_multiframe import VoxelBackBone8xMultiFrame
from .spconv_unet import UNetV2

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelResBackBone8xMultiFrame': VoxelResBackBone8x_multiframe,
    # 'VoxelResBackBone8xMultiFrame_ver2': VoxelResBackBone8x_multiframe_ver2,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal
}
