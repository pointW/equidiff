from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce, rearrange

from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import equi_diffpo.model.vision.crop_randomizer as dmvc

from equi_diffpo.model.vision.voxel_crop_randomizer import VoxelCropRandomizer
# from diffusion_policy.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.equi.equi_obs_encoder import InHandEncoder

# from diffusion_policy.model.equi.equi_conditional_unet1d_2 import D4ConditionalUnet1D
from equi_diffpo.model.vision.voxel_rot_randomizer import VoxelRotRandomizer

class CNNVoxelEncoder58(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128):
        super().__init__()
        self.obs_channel = obs_channel
        self.conv = torch.nn.Sequential(
            # 58
            torch.nn.Conv3d(obs_channel, n_out // 16, kernel_size=3, padding=0),
            # 56
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(2),
            # 28
            torch.nn.Conv3d(n_out // 16, n_out // 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(n_out // 8, n_out // 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(2),
            # 14
            torch.nn.Conv3d(n_out // 8, n_out // 4, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            # 12
            torch.nn.Conv3d(n_out // 4, n_out // 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(2),
            # 6
            torch.nn.Conv3d(n_out // 4, n_out // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(n_out // 2, n_out // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(2),
            # 3
            torch.nn.Conv3d(n_out // 2, n_out, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            # 1x1
        )

    def forward(self, x):
        return self.conv(x)
    
class ObsEncVoxel(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(4, 64, 64, 64),
        crop_shape=(64, 64, 64),
        n_hidden=128,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        if crop_shape[0] == 58:
            self.enc_obs = CNNVoxelEncoder58(obs_channel, self.n_hidden)
        else:
            raise NotImplementedError
        self.enc_ih = InHandEncoder(self.n_hidden).to(self.device)

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

    def forward(self, nobs):
        ee_pos = nobs["robot0_eef_pos"]
        obs = nobs["voxels"]
        ih = nobs["robot0_eye_in_hand_image"]
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
        enc_out = self.enc_obs(obs).reshape(batch_size * t, -1)  # b d
        ih_out = self.enc_ih(ih).reshape(batch_size * t, -1)
        features = torch.cat(
            [
                enc_out,
                ih_out,
                ee_pos,
                ee_quat,
                ee_q,
            ],
            dim=1,
        )
        return rearrange(features, "(b t) d -> b t d", b=batch_size)

class DiffusionUNetPolicyVoxel(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # image
        crop_shape=(58, 58, 58),
        # arch
        enc_n_hidden=64,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        rot_aug=True,
        color=True,
        depth=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        if color and depth:
            obs_channel = 4
        elif color:
            obs_channel = 3
        elif depth:
            obs_channel = 1

        self.enc = ObsEncVoxel(obs_shape=(obs_channel, 64, 64, 64), crop_shape=crop_shape, n_hidden=enc_n_hidden)

        obs_feature_dim = enc_n_hidden * 2 + 9
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.diff = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim, obs_dim=0, max_n_obs_steps=n_obs_steps, fix_obs_steps=True, action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = VoxelRotRandomizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug

        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], eps: float
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=weight_decay, lr=learning_rate, betas=betas, eps=eps
        )
        return optimizer

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.diff
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        obs_dict['voxels'][:, :, 1:] /= 255.0
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # condition through global feature
        # this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond, **self.kwargs
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        # reshape B, T, ... to B*T
        # this_nobs = dict_apply(nobs,
        #     lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.diff(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
