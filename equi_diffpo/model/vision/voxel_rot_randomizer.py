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

class VoxelRotRandomizer(nn.Module):
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


    # def forward(self, nobs, naction):
    #     """
    #     Randomly rotates the inputs if in training mode.
    #     Keeps inputs unchanged if in evaluation mode.

    #     Args:
    #         inputs (torch.Tensor): input tensors

    #     Returns:
    #         torch.Tensor: rotated or unrotated tensors based on the mode
    #     """
    #     if self.training:
    #         obs = nobs["agentview_voxel"]
    #         batch_size = obs.shape[0]
    #         T = obs.shape[1]
    #         C = obs.shape[2]

    #         angles = torch.rand(batch_size) * 2 * np.pi - np.pi
    #         rotation_matrix = torch.zeros((batch_size, 3, 3), device=obs.device)
    #         rotation_matrix[:, 2, 2] = 1
    #         for i, angle in enumerate(angles):
    #             # with a 1/64 probability, no rotation would be applied
    #             if np.random.random() < 1/64:
    #                 angle = 0
    #             rotation_matrix[i, :2, :] = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
    #                                                       [math.sin(angle), math.cos(angle), 0]], device=obs.device)
            
    #         obs = rearrange(obs, "b t c l h w -> b (t c l) h w")
    #         grid = F.affine_grid(rotation_matrix[:, :2], obs.size(), align_corners=True)
    #         rotated_obs = F.grid_sample(obs, grid, align_corners=True)
    #         rotated_obs = rearrange(rotated_obs, "b (t c l) h w -> b t c l h w", c=C, t=T)
    #         nobs["agentview_voxel"] = rotated_obs

    #         pos = nobs["robot0_eef_pos"]
    #         quat = nobs["robot0_eef_quat"]
    #         rot = self.tf.forward(quat)
    #         pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
    #         rot = rotation_matrix.unsqueeze(1) @ rot
    #         quat = self.tf.inverse(rot)
    #         pos = torch.clip(pos, -1, 1)
    #         nobs["robot0_eef_pos"] = pos
    #         nobs["robot0_eef_quat"] = quat

    #         naction[:, :, 0:3] = (rotation_matrix @ naction[:, :, 0:3].permute(0, 2, 1)).permute(0, 2, 1)
    #         naction[:, :, [3, 6]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [3, 6]].permute(0, 2, 1)).permute(0, 2, 1)
    #         naction[:, :, [4, 7]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [4, 7]].permute(0, 2, 1)).permute(0, 2, 1)
    #         naction[:, :, [5, 8]] = (rotation_matrix[:, :2, :2] @ naction[:, :, [5, 8]].permute(0, 2, 1)).permute(0, 2, 1)
    #         naction = torch.clip(naction, -1, 1)
            
    #         # nobs2 = deepcopy(nobs)
    #         # obs = nobs2["agentview_voxel"]
    #         # batch_size = obs.shape[0]
    #         # T = obs.shape[1]
    #         # C = obs.shape[2]
    #         # # angle = random.uniform(self.min_angle, self.max_angle)
    #         # angle = 90
    #         # angle_rad = math.radians(angle)
    #         # rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
    #         #                                 [math.sin(angle_rad), math.cos(angle_rad), 0],
    #         #                                 [0, 0, 1]]).to(obs.device)

    #         # obs = rearrange(obs, "b t c l h w -> (b t) (c l) h w")
    #         # rotated_obs = TF.rotate(obs, angle)
    #         # rotated_obs = rearrange(rotated_obs, "(b t) (c l) h w -> b t c l h w", b=batch_size, c=C)

    #         # nobs2["agentview_voxel"] = rotated_obs

    #         # pos = nobs2["robot0_eef_pos"]
    #         # quat = nobs2["robot0_eef_quat"]
    #         # pos = rearrange(pos, "b t d -> (b t) d")
    #         # quat = rearrange(quat, "b t d -> (b t) d")
    #         # rot = self.tf.forward(quat)
    #         # pos = (rotation_matrix @ pos.T).T
    #         # rot = rotation_matrix @ rot
    #         # quat = self.tf.inverse(rot)
    #         # pos = rearrange(pos, "(b t) d -> b t d", b=batch_size)
    #         # quat = rearrange(quat, "(b t) d -> b t d", b=batch_size)
    #         # nobs2["robot0_eef_pos"] = pos
    #         # nobs2["robot0_eef_quat"] = quat

    #         # naction = rearrange(naction, "b t d -> (b t) d")
    #         # naction[:, 0:3] = (rotation_matrix @ naction[:, 0:3].T).T
    #         # naction[:, [3, 6]] = (rotation_matrix[:2, :2] @ naction[:, [3, 6]].T).T
    #         # naction[:, [4, 7]] = (rotation_matrix[:2, :2] @ naction[:, [4, 7]].T).T
    #         # naction[:, [5, 8]] = (rotation_matrix[:2, :2] @ naction[:, [5, 8]].T).T
    #         # naction = rearrange(naction, "(b t) d -> b t d", b=batch_size)
    #         # naction = torch.clip(naction, -1, 1)
    #     return nobs, naction
    
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
            T = obs.shape[1]
            C = obs.shape[2]

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

            obs = rearrange(obs, "b t c h w d -> b t c d w h")
            obs = torch.flip(obs, (3, 4))
            obs = rearrange(obs, "b t c d w h -> b (t c d) w h")
            grid = F.affine_grid(rotation_matrix[:, :2], obs.size(), align_corners=True)
            rotated_obs = F.grid_sample(obs, grid, align_corners=True, mode='nearest')
            rotated_obs = rearrange(rotated_obs, "b (t c d) w h -> b t c d w h", c=C, t=T)
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

