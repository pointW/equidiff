import torch
import torch.nn as nn

class VoxelCropRandomizer(nn.Module):
    def __init__(
        self,
        crop_depth, 
        crop_height, 
        crop_width, 
    ):
        super().__init__()
        self.crop_depth = crop_depth
        self.crop_height = crop_height
        self.crop_width = crop_width
        
    def forward(self, voxels):
        B, C, D, H, W = voxels.shape
        if self.training:
            cropped_voxel = []
            for i in range(B):
                d_start = torch.randint(0, D-self.crop_depth+1, [1])[0]
                h_start = torch.randint(0, H-self.crop_height+1, [1])[0]
                w_start = torch.randint(0, W-self.crop_width+1, [1])[0]
                cropped_voxel.append(voxels[i, 
                                            :, 
                                            d_start:d_start+self.crop_depth,
                                            h_start:h_start+self.crop_height,
                                            w_start:w_start+self.crop_width,
                                            ])
            return torch.stack(cropped_voxel, 0)
        else:
            voxels = voxels[:, 
                            :,
                            D//2 - self.crop_depth//2: D//2 + self.crop_depth//2,
                            H//2 - self.crop_height//2: H//2 + self.crop_height//2,
                            W//2 - self.crop_width//2: W//2 + self.crop_width//2,
                            ]
            return voxels
