from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D

class EquiDiffusionUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        cos2 = conv_out[:, 4:5]
        sin2 = conv_out[:, 5:6]
        cos3 = conv_out[:, 6:7]
        sin3 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        rot = act[:, 3:9]
        g = act[:, 9:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                rot[:, 0].reshape(batch_size, 1),
                rot[:, 3].reshape(batch_size, 1),
                rot[:, 1].reshape(batch_size, 1),
                rot[:, 4].reshape(batch_size, 1),
                rot[:, 2].reshape(batch_size, 1),
                rot[:, 5].reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out
 
class EquiDiffusionUNetSE2(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):

        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            2 * [self.group.irrep(1)] # 4
            + 2 * [self.group.trivial_repr], # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        z = conv_out[:, 4:5]
        g = conv_out[:, 5:6]

        action = torch.cat((xy, z, cos1, sin1, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        cos = act[:, 3:4]
        sin = act[:, 4:5]
        g = act[:, 5:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                cos.reshape(batch_size, 1),
                sin.reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out