import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange, repeat
import math
from copy import deepcopy

from equi_diffpo.model.common.rotation_transformer import RotationTransformer

class VoxelRotRandomizerRel(nn.Module):
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
        self.quat_to_matrix = RotationTransformer('quaternion', 'matrix')
        self.axisangle_to_matrix = RotationTransformer('axis_angle', 'matrix')
    

    def forward(self, nobs, naction: torch.Tensor):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training:
            obs = nobs["voxels"]
            pos = nobs["robot0_eef_pos"]
            # x, y, z, w -> w, x, y, z
            quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
            batch_size = obs.shape[0]
            To = obs.shape[1]
            C = obs.shape[2]
            Ta = naction.shape[1]

            for i in range(1000):
                angles = torch.rand(batch_size) * 2 * np.pi - np.pi
                rotation_matrix = torch.zeros((batch_size, 3, 3), device=obs.device)
                rotation_matrix[:, 2, 2] = 1
                # construct rotation matrix
                angles[torch.rand(batch_size) < 1/64] = 0
                rotation_matrix[:, 0, 0] = torch.cos(angles)
                rotation_matrix[:, 0, 1] = -torch.sin(angles)
                rotation_matrix[:, 1, 0] = torch.sin(angles)
                rotation_matrix[:, 1, 1] = torch.cos(angles)
                # rotating the xyz vector in action
                rotated_naction = naction.clone()
                expanded_rotation_matrix = repeat(rotation_matrix, 'b d1 d2 -> b t d1 d2', t=Ta)
                rotated_naction[:, :, :3] = (expanded_rotation_matrix @ naction[:, :, :3].unsqueeze(-1)).squeeze(-1)
                # rotating the axis angle rotation vector in action
                axis_angle = rotated_naction[:, :, 3:6]
                m = self.axisangle_to_matrix.forward(axis_angle)
                rotation_matrix_inv = rotation_matrix.transpose(1, 2)
                expanded_rotation_matrix_inv = repeat(rotation_matrix_inv, 'b d1 d2 -> b t d1 d2', t=Ta)
                gm = expanded_rotation_matrix @ m @ expanded_rotation_matrix_inv
                g_axis_angle = self.axisangle_to_matrix.inverse(gm)
                rotated_naction[:, :, 3:6] = g_axis_angle
                # rotating state pos and quat
                rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
                rot = self.quat_to_matrix.forward(quat)
                rotated_rot = rotation_matrix.unsqueeze(1) @ rot
                rotated_quat = self.quat_to_matrix.inverse(rotated_rot)

                if rotated_pos.min() >= -1 and rotated_pos.max() <= 1:
                    break
            if i == 999:
                return nobs, naction

            obs = rearrange(obs, "b t c h w d -> b t c d w h")
            obs = torch.flip(obs, (3, 4))
            obs = rearrange(obs, "b t c d w h -> b (t c d) w h")
            grid = F.affine_grid(rotation_matrix[:, :2], obs.size(), align_corners=True)
            rotated_obs = F.grid_sample(obs, grid, align_corners=True, mode='nearest')
            rotated_obs = rearrange(rotated_obs, "b (t c d) w h -> b t c d w h", c=C, t=To)
            rotated_obs = torch.flip(rotated_obs, (3, 4))
            rotated_obs = rearrange(rotated_obs, "b t c d w h -> b t c h w d")

            nobs["voxels"] = rotated_obs
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

