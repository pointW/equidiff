from equi_diffpo.dataset.base_dataset import LinearNormalizer
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset, normalizer_from_stat
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_symmetric_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_range_symmetric_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import numpy as np

class RobomimicReplayImageSymDataset(RobomimicReplayImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100
        ):
        super().__init__(
            shape_meta,
            dataset_path,
            horizon,
            pad_before,
            pad_after,
            n_obs_steps,
            abs_action,
            rotation_rep,
            use_legacy_normalizer,
            use_cache,
            seed,
            val_ratio,
            n_demo
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

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos'):
                this_normalizer = get_range_symmetric_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.find('bbox') > -1:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        normalizer['pos_vecs'] = get_identity_normalizer_from_stat({'min': -1 * np.ones([10, 2], np.float32), 'max': np.ones([10, 2], np.float32)})
        normalizer['crops'] = get_image_range_normalizer()

        return normalizer

