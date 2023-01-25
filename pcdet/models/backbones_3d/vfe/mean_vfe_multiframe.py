import torch

from .vfe_template import VFETemplate


class MeanVFEMulti(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        mean = []
        for ind, (vf, vnp) in enumerate(zip(voxel_features, voxel_num_points)):
            points_mean = vf[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(vnp.view(-1, 1), min=1.0).type_as(vf)
            points_mean = points_mean / normalizer
            mean.append(points_mean.contiguous())
            
            
        batch_dict['voxel_features'] = mean
        del mean

        return batch_dict
