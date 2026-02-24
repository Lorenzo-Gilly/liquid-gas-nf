import torch
from torch import nn
from einops import rearrange
from normalizing_flow.splines import rational_quadratic_spline

from tools.util import get_target_indices

class parameter_equivariant_network(nn.Module):

    def __init__(self, input_size, output_size, device, transformer_args={"depth" : 1, "dim" : 128}, n_freqs=8):
        super(parameter_equivariant_network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.n_freqs = n_freqs
        self.freqs = torch.arange(self.n_freqs, device=device).reshape(1, 1, -1) + 1

        self.lin_in = nn.Linear(self.input_size * 2 * self.n_freqs, transformer_args["dim"])
        
        self.transformer_encoder = nn.TransformerEncoder(
                                        nn.TransformerEncoderLayer(
                                            d_model = transformer_args["dim"], 
                                            nhead = transformer_args["dim"]//64, 
                                            dim_feedforward = transformer_args["dim"]*4, 
                                            batch_first = True, 
                                            norm_first = True, 
                                            dropout=0.0
                                        ), 
                                        transformer_args["depth"]
                                    )
        
        self.lin_out = nn.Linear(transformer_args["dim"], self.output_size)


    def forward(self, x):
        
        x = x.reshape(x.shape[0], -1, self.input_size)

        # Circular encoder (input must be between -1 and 1)
        cos_enc = torch.cos(self.freqs * torch.pi * x.unsqueeze(-1))
        sin_enc = torch.sin(self.freqs * torch.pi * x.unsqueeze(-1))

        x = torch.cat([
                cos_enc.view(x.shape[0], -1, self.input_size * self.n_freqs),
                sin_enc.view(x.shape[0], -1, self.input_size * self.n_freqs)
            ], dim=-1)

        x = self.lin_in(x)
        x = self.transformer_encoder(x)
        x = self.lin_out(x)

        x = x.reshape(x.shape[0], -1)

        return x
    

class RQS_coupling_block(nn.Module):

    def __init__(self, target_coordinates, n_particles, dimensions, device, n_bins=8, left=-1, right=1, bottom=-1, top=1):
        super(RQS_coupling_block, self).__init__()

        self.n_particles = n_particles
        self.dimensions = dimensions

        self.n_bins = n_bins
        self.left   = left
        self.right  = right
        self.bottom = bottom
        self.top    = top

        in_dim, out_dim = dimensions-len(target_coordinates), len(target_coordinates)
        identity_indices, transformed_indices = get_target_indices(target_coordinates, n_particles, dimensions)

        self.identity_indices = identity_indices
        self.transformed_indices = transformed_indices

        # Choosing the network to get rational quadratice spline parameters
        self.network = parameter_equivariant_network(in_dim, out_dim * (3 * self.n_bins), device=device)


    def forward(self, x, inverse=False):

        # Split input
        x_identity = x[:, self.identity_indices]
        x_transformed = x[:, self.transformed_indices]

        # Rational Quadratic splines parameter via equivariant transformer
        parameters = self.network(x_identity) # Parameters of the transformation are function of the untransformed input
        parameters = rearrange(parameters, "b (d p) -> b d p", d = len(self.transformed_indices)) # p = (widths, heights, slopes) * n_bins, d = n_particles

        widths = parameters[:, :, :self.n_bins]
        heights = parameters[:, :, self.n_bins:2*self.n_bins]
        slopes = parameters[:, :, 2*self.n_bins:]
        # Make spline periodic
        slopes = torch.cat([slopes, slopes[..., [0]]], dim=-1)

        # Part of input transformed through a function of the untransformed input xt = (f_xi)(xt)
        x_transformed, part_log_det = rational_quadratic_spline(
                                        x_transformed, 
                                        widths, heights, slopes, 
                                        inverse=inverse, 
                                        left=self.left, right=self.right,
                                        bottom=self.bottom, top=self.top,
                                        enable_identity_init=False)

        x[:, self.transformed_indices] = x_transformed

        return x, part_log_det.sum(dim=-1, keepdim=True)