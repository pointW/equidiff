from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from equi_diffpo.model.equi.equi_obs_encoder import EquivariantObsEnc
from equi_diffpo.model.equi.equi_conditional_unet1d import EquiDiffusionUNetSE2
from equi_diffpo.model.vision.rot_randomizer import RotRandomizer


class DiffusionEquiUNetCNNEncPolicySE2(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            # arch
            N=8,
            enc_n_hidden=64,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = 6
        obs_shape_meta = shape_meta['obs']
        
        self.enc = EquivariantObsEnc(
            obs_shape=obs_shape_meta['agentview_image']['shape'], 
            crop_shape=crop_shape, 
            n_hidden=enc_n_hidden, 
            N=N)
        
        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        self.diff = EquiDiffusionUNetSE2(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
        )

        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim

        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.canonical_to_world = torch.inverse(torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        self.sixd_to_matrix = RotationTransformer('rotation_6d', 'matrix')
        self.sixd_to_euler = RotationTransformer('rotation_6d', 'euler_angles', to_convention='ZYX')

    def getRelative6D(self, rot6d):
        gTw = self.sixd_to_matrix.forward(rot6d)
        gTc = torch.matmul(gTw, self.canonical_to_world.to(self.device))
        return self.sixd_to_matrix.inverse(gTc)
    
    def getAbsolute6D(self, rot6d):
        gTc = self.sixd_to_matrix.forward(rot6d)
        cTw = torch.inverse(self.canonical_to_world.to(self.device))
        gTw = torch.matmul(gTc, cTw)
        return self.sixd_to_matrix.inverse(gTw)

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            weight_decay: float, 
            learning_rate: float, 
            betas: Tuple[float, float],
            eps: float
        ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=weight_decay, lr=learning_rate, betas=betas, eps=eps
        )
        return optimizer
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.diff
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        cos = naction_pred[:, :, 3:4]
        sin = naction_pred[:, :, 4:5]

        naction_pred = torch.cat((naction_pred[:, :, :3], cos, -sin, torch.zeros_like(cos), sin, cos, torch.zeros_like(cos), naction_pred[:, :, 5:]), dim=2)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        abs_6d = self.getAbsolute6D(action_pred[:, :, 3:9])
        action_pred[:, :, 3:9] = abs_6d

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        actions = batch['action'].clone()
        rel_6d = self.getRelative6D(actions[:, :, 3:9])

        actions[:, :, 3:9] = rel_6d
        nactions = self.normalizer['action'].normalize(actions)
        sixd = nactions[:, :, 3:9]
        euler = self.sixd_to_euler.forward(sixd)
        rotz = euler[:, :, 0:1]
        cos = torch.cos(rotz)
        sin = torch.sin(rotz)
        nactions = torch.cat([nactions[:, :, :3], cos, sin, nactions[:, :, 9:]], dim=2)

        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory
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
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.diff(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss