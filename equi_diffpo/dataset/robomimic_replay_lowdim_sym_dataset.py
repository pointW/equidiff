from typing import Dict, List
import torch
import numpy as np
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.dataset.base_dataset import LinearNormalizer
from equi_diffpo.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset, normalizer_from_stat
from equi_diffpo.common.normalize_util import robomimic_abs_action_only_symmetric_normalizer_from_stat
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_symmetric_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class RobomimicReplayLowdimSymDataset(RobomimicReplayLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_demo=100
        ):
        super().__init__(
            dataset_path,
            horizon,
            pad_before,
            pad_after,
            obs_keys,
            abs_action,
            rotation_rep,
            use_legacy_normalizer,
            seed,
            val_ratio,
            max_train_episodes,
            n_demo,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                raise NotImplementedError
            else:
                this_normalizer = robomimic_abs_action_only_symmetric_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # aggregate obs stats
        obs_stat = array_to_stats(self.replay_buffer['obs'])


        normalizer['obs'] = normalizer_from_stat(obs_stat)
        return normalizer
