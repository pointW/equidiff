import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange
import math

from equi_diffpo.model.common.rotation_transformer import RotationTransformer

class RotRandomizer(nn.Module):
    """
    Continuously and randomly rotate the input tensor during training.
    Does not rotate the tensor during evaluation.
    """
    
    def __init__(self, min_angle=-180, max_angle=180):
        """
        Args:
            min_angle (float): Minimum rotation angle.
            max_angle (float): Maximum rotation angle.
        """
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.tf = RotationTransformer('quaternion', 'matrix')


    def forward(self, nobs, naction):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training:
            obs = nobs["agentview_image"]
            pos = nobs["robot0_eef_pos"]
            # x, y, z, w -> w, x, y, z
            quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
            batch_size = obs.shape[0]
            T = obs.shape[1]

            for i in range(1000):
                angles = torch.rand(batch_size) * 2 * np.pi - np.pi
                rotation_matrix = torch.zeros((batch_size, 3, 3), device=obs.device)
                rotation_matrix[:, 2, 2] = 1

                angles[torch.rand(batch_size) < 1/64] = 0
                rotation_matrix[:, 0, 0] = torch.cos(angles)
                rotation_matrix[:, 0, 1] = -torch.sin(angles)
                rotation_matrix[:, 1, 0] = torch.sin(angles)
                rotation_matrix[:, 1, 1] = torch.cos(angles)

                rotated_naction = naction.clone()
                rotated_naction[:, :, 0:3] = (rotation_matrix @ naction[:, :, 0:3].permute(0, 2, 1)).permute(0, 2, 1)
                rotated_naction[:, :, [3, 6]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [3, 6]].permute(0, 2, 1)).permute(0, 2, 1)
                rotated_naction[:, :, [4, 7]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [4, 7]].permute(0, 2, 1)).permute(0, 2, 1)
                rotated_naction[:, :, [5, 8]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [5, 8]].permute(0, 2, 1)).permute(0, 2, 1)

                rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
                rot = self.tf.forward(quat)
                rotated_rot = rotation_matrix.unsqueeze(1) @ rot
                rotated_quat = self.tf.inverse(rotated_rot)

                if rotated_pos.min() >= -1 and rotated_pos.max() <= 1 and rotated_naction[:, :, :2].min() > -1 and rotated_naction[:, :, :2].max() < 1:
                    break
            if i == 999:
                return nobs, naction

            obs = rearrange(obs, "b t c h w -> b (t c) h w")
            grid = F.affine_grid(rotation_matrix[:, :2], obs.size(), align_corners=True)
            rotated_obs = F.grid_sample(obs, grid, align_corners=True, mode='bilinear')
            rotated_obs = rearrange(rotated_obs, "b (t c) h w -> b t c h w", b=batch_size, t=T)

            nobs["agentview_image"] = rotated_obs
            nobs["robot0_eef_pos"] = rotated_pos
            # w, x, y, z -> x, y, z, w
            nobs["robot0_eef_quat"] = rotated_quat[:, :, [1, 2, 3, 0]]
            naction = rotated_naction

        return nobs, naction

    def __repr__(self):
        """Pretty print the network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(self.min_angle, self.max_angle)
        return msg

