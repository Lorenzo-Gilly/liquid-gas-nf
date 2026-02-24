import numpy as np
import torch
from torch import nn

from normalizing_flow.base import base_generator

class flow_assembler(base_generator):

    def __init__(self, prior, posterior, blocks, prior_sided_transformation_layers=[], post_sided_transformation_layers=[], device=None):
        super(flow_assembler, self).__init__(prior, posterior, device)

        if device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_avail() else torch.device("cpu")
        else:
            self.device = device

        self.prior_sided_transformation_layers = nn.ModuleList(prior_sided_transformation_layers)
        self.posterior_sided_transformation_layers = nn.ModuleList(post_sided_transformation_layers)

        self.n_prior_sided_tl = len(prior_sided_transformation_layers)
        self.n_post_sided_tl = len(post_sided_transformation_layers)

        self.blocks = nn.ModuleList(blocks)

        self.apply(self.init_weights)

    def init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
        
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=0.01)


    def F_xz(self, x):

        z, logdetJ_xz = x.clone(), x.new_zeros(x.shape[0], 1)

        for t in range(self.n_post_sided_tl):

            z, part_logdetJ_data2network = self.posterior_sided_transformation_layers[t].F_data2network(z)
            logdetJ_xz += part_logdetJ_data2network

        for block in reversed(self.blocks):

            z, part_logdetJ_xz = block(z, inverse=True)
            logdetJ_xz += part_logdetJ_xz

        for t in reversed(range(self.n_prior_sided_tl)):

            z, part_logdetJ_network2data = self.prior_sided_transformation_layers[t].F_network2data(z)
            logdetJ_xz += part_logdetJ_network2data
        
        return z, logdetJ_xz
    

    def F_zx(self, z):

        x, logdetJ_zx = z.clone(), z.new_zeros(z.shape[0], 1)

        for t in range(self.n_prior_sided_tl):

            x, part_logdetJ_data2network = self.prior_sided_transformation_layers[t].F_data2network(x)
            logdetJ_zx += part_logdetJ_data2network

        for block in self.blocks:

            x, part_logdetJ_zx = block(x, inverse=False)
            logdetJ_zx += part_logdetJ_zx

        for t in reversed(range(self.n_post_sided_tl)):

            x, part_logdetJ_network2data = self.posterior_sided_transformation_layers[t].F_network2data(x)
            logdetJ_zx += part_logdetJ_network2data

        return x, logdetJ_zx
    

    # This method computes the T2S loss (negative log-likelihood)
    def loss_xz(self, x, beta_source, beta_target, energy_x=None, min_val=None, max_val=None):
        
        z, logdetJ_xz = self.F_xz(x)

        logp_xz = -beta_source*self.prior.energy(z)
        if min_val is not None or max_val is not None:
            logp_xz = torch.clamp(logp_xz, min=min_val, max=max_val)
        
        logw_xz = None
        if not self.training:
            if energy_x is None:
                energy_x = self.posterior.energy(x)
            logp_x = -beta_target*energy_x
            logw_xz = (logp_xz - logp_x + logdetJ_xz).squeeze(-1)

        return -(logp_xz + logdetJ_xz).mean(), logw_xz
        
        
    # This method computes the S2T loss (kullback-leibler divergence)
    def loss_zx(self, z, beta_target, beta_source, energy_z=None, min_val=None, max_val=None):
        
        x, logdetJ_zx = self.F_zx(z)

        logp_zx = -beta_target*self.posterior.energy(x)
        if min_val is not None or max_val is not None:
            logp_zx = torch.clamp(logp_zx, min=min_val, max=max_val)

        logw_zx = None
        if not self.training:
            if energy_z is None:
                energy_z = self.prior.energy(z)
            logp_z = -beta_source*energy_z
            logw_zx = (logp_zx - logp_z + logdetJ_zx).squeeze(-1)

        return -(logp_zx + logdetJ_zx).mean(), logw_zx