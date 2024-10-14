from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.common.rotation_transformer import RotationTransformer


class EquiDiffusionUNetVel(torch.nn.Module):
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

        self.p = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, 0]
        ]).float()
        self.p_inv = torch.linalg.inv(self.p)
        self.axisangle_to_matrix = RotationTransformer('axis_angle', 'matrix')

    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            1 * [self.group.irrep(2)] # 2
            + 3 * [self.group.irrep(1)] # 6
            + 5 * [self.group.trivial_repr], # 5
        )
    
    # matrix
    def getOutput(self, conv_out):
        rho2 = conv_out[:, 0:2]
        xy = conv_out[:, 2:4]
        rho11 = conv_out[:, 4:6]
        rho12 = conv_out[:, 6:8]
        rho01 = conv_out[:, 8:9]
        rho02 = conv_out[:, 9:10]
        rho03 = conv_out[:, 10:11]
        z = conv_out[:, 11:12]
        g = conv_out[:, 12:13]

        v = torch.cat((rho01, rho02, rho03, rho11, rho12, rho2), dim=1)
        m = torch.matmul(self.p_inv.to(conv_out.device), v.reshape(-1, 9, 1)).reshape(-1, 9)

        action = torch.cat((xy, z, m, g), dim=1)
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        xy = act[:, 0:2]
        z = act[:, 2:3]
        m = act[:, 3:12]
        g = act[:, 12:]

        v = torch.matmul(self.p.to(act.device), m.reshape(-1, 9, 1)).reshape(-1, 9)

        cat = torch.cat(
            (
                v[:, 7:9].reshape(batch_size, 2),
                xy.reshape(batch_size, 2),
                v[:, 5:7].reshape(batch_size, 2),
                v[:, 3:5].reshape(batch_size, 2),
                v[:, 0:3].reshape(batch_size, 3),
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
