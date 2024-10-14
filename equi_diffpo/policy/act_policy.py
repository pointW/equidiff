import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from equi_diffpo.model.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.common.pytorch_util import dict_apply
import torch
from typing import Dict, Tuple
import numpy as np

class ACTPolicyWrapper(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 max_timesteps: int,
                 temporal_agg: bool,
                 n_envs: int,
                 horizon: int=10,
                 ):
        super().__init__()
        action_dim = 10
        lr = 5e-5
        lr_backbone = 5e-5
        chunk_size = horizon
        kl_weight = 10
        hidden_dim = 512
        dim_feedforward = 3200
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': lr,
                        'num_queries': chunk_size,
                        'kl_weight': kl_weight,
                        'hidden_dim': hidden_dim,
                        'dim_feedforward': dim_feedforward,
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': ['agentview_image', 'robot0_eye_in_hand_image'],

                        "weight_decay": 1e-4,
                        "dilation": False,
                        "position_embedding": "sine",
                        "dropout": 0.1,
                        "pre_norm": False,
                        "masks": False,
                        }
        self.model = ACTPolicy(policy_config)
        self.optimizer = self.model.configure_optimizers()
        self.normalizer = LinearNormalizer()

        self.quat_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.num_queries = policy_config['num_queries']
        self.query_frequency = 1
        self.temporal_agg = temporal_agg
        self.max_timesteps = max_timesteps
        self.action_dim = action_dim

        self.n_envs = n_envs

        self.all_time_actions = torch.zeros([self.n_envs, self.max_timesteps, self.max_timesteps+self.num_queries, self.action_dim]).to(self.device)
        self.t = 0

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # nobs_dict = self.normalizer(obs_dict)
        nobs_dict = dict_apply(obs_dict, lambda x: x[:,0,...])
        # sixd = self.quat_to_sixd.forward(nobs_dict['robot0_eef_quat'])
        qpos = torch.cat([nobs_dict['robot0_eef_pos'], nobs_dict['robot0_eef_quat'], nobs_dict['robot0_gripper_qpos']], dim=1)
        image = torch.stack([nobs_dict['agentview_image'], nobs_dict['robot0_eye_in_hand_image']], dim=1)

        if self.temporal_agg:
            if self.t % self.query_frequency == 0:
                all_actions = self.model(qpos, image)
                self.all_actions = all_actions
            else:
                all_actions = self.all_actions
            self.all_time_actions[:, self.t, self.t:self.t+self.num_queries] = all_actions
            actions_for_curr_step = self.all_time_actions[:, :, self.t]

            actions_populated = torch.all(actions_for_curr_step != 0, axis=2)

            raw_actions = []
            for i in range(self.n_envs):
                populated_actions = actions_for_curr_step[i, actions_populated[i]]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(populated_actions)))
                exp_weights /= exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                raw_action = (populated_actions * exp_weights).sum(dim=0, keepdim=True)
                raw_actions.append(raw_action)
            raw_action = torch.cat(raw_actions, dim=0)


            # actions_for_curr_step = actions_for_curr_step[actions_populated].reshape(self.n_envs, -1, 10)

            # # actions_for_curr_step2 = []
            # # for i in range(2):
            # #     actions_for_curr_step2.append(actions_for_curr_step[i:i+1][actions_populated[i:i+1]])
            # # actions_for_curr_step2 = torch.stack(actions_for_curr_step2, dim=0)

            # k = 0.01
            # exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[1]))
            # exp_weights = exp_weights / exp_weights.sum()
            # exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            # raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)

            action = self.normalizer['action'].unnormalize(raw_action)
            # (B, Da)
            result = {
                'action': action[:,None,:] # (B, 1, Da)
            }

        else:
            raw_action = self.model(qpos, image)
            action = self.normalizer['action'].unnormalize(raw_action)
            result = {
                'action': action # (B, 1, Da)
            }
        self.t += 1
        return result
    
    def reset(self):
        self.all_time_actions = torch.zeros([self.n_envs, self.max_timesteps, self.max_timesteps+self.num_queries, self.action_dim]).to(self.device)
        self.t = 0

    def compute_loss(self, batch):
        # nobs_dict = self.normalizer.normalize(batch['obs'])
        nobs_dict = batch['obs']
        nactions = self.normalizer['action'].normalize(batch['action'])
        nobs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        # sixd = self.quat_to_sixd.forward(nobs_dict['robot0_eef_quat'])
        qpos = torch.cat([nobs_dict['robot0_eef_pos'], nobs_dict['robot0_eef_quat'], nobs_dict['robot0_gripper_qpos']], dim=1)
        image = torch.stack([nobs_dict['agentview_image'], nobs_dict['robot0_eye_in_hand_image']], dim=1)

        forward_dict = self.model(qpos, image, nactions, torch.zeros([*nactions.shape[:2]]).bool().to(self.device))
        return forward_dict['loss']
