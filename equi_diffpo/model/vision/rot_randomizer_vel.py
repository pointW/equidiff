import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from einops import rearrange
import math

from equi_diffpo.model.common.rotation_transformer import RotationTransformer

class RotRandomizerVel(nn.Module):
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


    def forward(self, nobs, naction):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training and np.random.random() > 1/64:
            obs = nobs["agentview_image"]
            batch_size = obs.shape[0]
            
            angle = random.uniform(self.min_angle, self.max_angle)
            angle_rad = math.radians(angle)
            rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                            [math.sin(angle_rad), math.cos(angle_rad), 0],
                                            [0, 0, 1]]).to(obs.device)


            obs = rearrange(obs, "b t c h w -> (b t) c h w")
            rotated_obs = TF.rotate(obs, angle)
            rotated_obs = rearrange(rotated_obs, "(b t) c h w -> b t c h w", b=batch_size)
            nobs["agentview_image"] = rotated_obs

            if "crops" in nobs:
                crops = nobs["crops"]
                n_crop = crops.shape[2]
                crops = rearrange(crops, "b t n c h w -> (b t n) c h w")
                crops = TF.rotate(crops, angle)
                crops = rearrange(crops, "(b t n) c h w -> b t n c h w", b=batch_size, n=n_crop)
                nobs["crops"] = crops

            if "pos_vecs" in nobs:
                pos_vecs = nobs["pos_vecs"]
                pos_vecs = rearrange(pos_vecs, "b t n d -> (b t n) d")
                pos_vecs = (rotation_matrix[:2, :2] @ pos_vecs.T).T
                pos_vecs = rearrange(pos_vecs, "(b t n) d -> b t n d", b=batch_size, n=n_crop)
                nobs["pos_vecs"] = pos_vecs

            pos = nobs["robot0_eef_pos"]
            quat = nobs["robot0_eef_quat"]
            pos = rearrange(pos, "b t d -> (b t) d")
            quat = rearrange(quat, "b t d -> (b t) d")
            rot = self.quat_to_matrix.forward(quat)
            pos = (rotation_matrix @ pos.T).T
            rot = rotation_matrix @ rot
            quat = self.quat_to_matrix.inverse(rot)
            pos = rearrange(pos, "(b t) d -> b t d", b=batch_size)
            quat = rearrange(quat, "(b t) d -> b t d", b=batch_size)
            nobs["robot0_eef_pos"] = pos
            nobs["robot0_eef_quat"] = quat

            naction = rearrange(naction, "b t d -> (b t) d")
            naction[:, 0:3] = (rotation_matrix @ naction[:, 0:3].T).T
            axis_angle = naction[:, 3:6]
            m = self.axisangle_to_matrix.forward(axis_angle)
            gm = rotation_matrix @ m @ torch.linalg.inv(rotation_matrix)
            g_axis_angle = self.axisangle_to_matrix.inverse(gm)
            naction[:, 3:6] = g_axis_angle

            naction = rearrange(naction, "(b t) d -> b t d", b=batch_size)
            naction = torch.clip(naction, -1, 1)
        return nobs, naction

    def __repr__(self):
        """Pretty print the network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(self.min_angle, self.max_angle)
        return msg

