from .SelfAttention import SelfAttentionTemplate
from .sa_compression import SelfAttentionCompression, SelfAttentionCompression_v2
from .voxel_fsa import VoxelContext3D_fsa
from .voxel_dsa import VoxelContext3D_dsa
from .video_swin_transformer import SwinTransformer3D


__all__ = {
    'SelfAttention':SelfAttentionTemplate,
    'SelfAttentionCompression': SelfAttentionCompression,
    'SelfAttentionCompression_v2': SelfAttentionCompression_v2,
    'VoxelContext3D_fsa': VoxelContext3D_fsa,
    'VoxelContext3D_dsa': VoxelContext3D_dsa,
    'SwinTransformer3D': SwinTransformer3D
}