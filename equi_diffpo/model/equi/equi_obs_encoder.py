import torch
from torchvision import models as vision_models
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange
from robomimic.models.base_nets import SpatialSoftmax
from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
import equi_diffpo.model.vision.crop_randomizer as dmvc
from equi_diffpo.model.equi.equi_encoder import EquivariantResEncoder76Cyclic, EquivariantVoxelEncoder58Cyclic, EquivariantVoxelEncoder64Cyclic
from equi_diffpo.model.vision.voxel_crop_randomizer import VoxelCropRandomizer
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class InHandEncoder(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=Identity)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)


class EquivariantObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.token_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])
        self.enc_obs = EquivariantResEncoder76Cyclic(obs_channel, self.n_hidden, initialize)
        self.enc_ih = InHandEncoder(self.n_hidden).to(self.device)
        self.enc_out = nn.Linear(
            nn.FieldType(
                self.group,
                n_hidden * [self.group.regular_repr] # agentview
                + n_hidden * [self.group.trivial_repr] # ih
                + 4 * [self.group.irrep(1)] # pos, rot
                + 3 * [self.group.trivial_repr], # gripper (2), z zpos
            ),
            self.token_type,
        )
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
        
    def forward(self, nobs):
        obs = nobs["agentview_image"]
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        ih = nobs["robot0_eye_in_hand_image"]
        # B, T, C, H, W
        batch_size = obs.shape[0]
        t = obs.shape[1]
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        obs = self.crop_randomizer(obs)
        ih = self.crop_randomizer(ih)
        ee_rot = self.get6DRotation(ee_quat)

        enc_out = self.enc_obs(obs).tensor.reshape(batch_size * t, -1)  # b d
        ih_out = self.enc_ih(ih)
        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]
        features = torch.cat(
            [
                enc_out,
                ih_out,
                pos_xy,
                # ee_rot is the first two rows of the rotation matrix (i.e., the rotation 6D repr.)
                # each column vector in the first two rows of the rotation 6d forms a rho1 vector
                ee_rot[:, 0:1],
                ee_rot[:, 3:4],
                ee_rot[:, 1:2],
                ee_rot[:, 4:5],
                ee_rot[:, 2:3],
                ee_rot[:, 5:6],
                pos_z,
                ee_q,
            ],
            dim=1
        )
        features = nn.GeometricTensor(features, self.enc_out.in_type)
        out = self.enc_out(features).tensor
        return rearrange(out, "(b t) d -> b t d", b=batch_size)
    
class EquivariantObsEncVoxel(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(4, 64, 64, 64),
        crop_shape=(64, 64, 64),
        n_hidden=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.token_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])
        if crop_shape[0] == 58:
            self.enc_obs = EquivariantVoxelEncoder58Cyclic(obs_channel, self.n_hidden, initialize)
        else:
            self.enc_obs = EquivariantVoxelEncoder64Cyclic(obs_channel, self.n_hidden, initialize)
        self.enc_ih = InHandEncoder(self.n_hidden).to(self.device)
        self.enc_out = nn.Linear(
            nn.FieldType(
                self.group,
                n_hidden * [self.group.regular_repr] # agentview
                + n_hidden * [self.group.trivial_repr] # ih
                + 4 * [self.group.irrep(1)] # pos, rot
                + 3 * [self.group.trivial_repr], # gripper (2), z zpos
            ),
            self.token_type,
        )
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        if crop_shape[0] == 58:
            self.voxel_crop_randomizer = VoxelCropRandomizer(
                crop_depth=crop_shape[0],
                crop_height=crop_shape[1],
                crop_width=crop_shape[2],
            )
        self.crop_shape = crop_shape

        self.ih_crop_randomizer = dmvc.CropRandomizer(
            input_shape=(3, 84, 84),
            crop_height=76,
            crop_width=76,
        )

    def get6DRotation(self, quat):
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]])    
    
    def forward(self, nobs):
        ee_pos = nobs["robot0_eef_pos"]
        ih = nobs["robot0_eye_in_hand_image"]
        obs = nobs["voxels"]
        obs = rearrange(obs, "b t c h w d -> b t c d w h")
        obs = torch.flip(obs, (3, 4))

        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        # B, T, C, H, W
        batch_size = obs.shape[0]
        t = obs.shape[1]
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        obs = rearrange(obs, "b t c h w l -> (b t) c h w l")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        if self.crop_shape[0] == 58:
            obs = self.voxel_crop_randomizer(obs)
        ih = self.ih_crop_randomizer(ih)
        ee_rot = self.get6DRotation(ee_quat)
        enc_out = self.enc_obs(obs).tensor.reshape(batch_size * t, -1)  # b d
        ih_out = self.enc_ih(ih)
        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]
        features = torch.cat(
            [
                enc_out,
                ih_out,
                pos_xy,
                # ee_rot is the first two rows of the rotation matrix (i.e., the rotation 6D repr.)
                # each column vector in the first two rows of the rotation 6d forms a rho1 vector
                ee_rot[:, 0:1],
                ee_rot[:, 3:4],
                ee_rot[:, 1:2],
                ee_rot[:, 4:5],
                ee_rot[:, 2:3],
                ee_rot[:, 5:6],
                pos_z,
                ee_q,
            ],
            dim=1
        )
        features = nn.GeometricTensor(features, self.enc_out.in_type)
        out = self.enc_out(features).tensor
        return rearrange(out, "(b t) d -> b t d", b=batch_size)