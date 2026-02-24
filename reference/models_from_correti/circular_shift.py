import numpy as np
import torch
from torch import nn

class circular_shift(nn.Module):

    def __init__(self, n_particles, dimensions, device, box_length=2):
        super(circular_shift, self).__init__()

        self.n_particles = n_particles
        self.dimensions = dimensions
        self.box_length = box_length

        # Circular shift: this parameter must be learnable
        self.circular_shift = nn.Parameter(torch.zeros(dimensions, device=device))


    def forward(self, x, inverse=False):
        
        # Circular shift in "box" [-L/2,L/2]
        x = x.view(-1, self.n_particles, self.dimensions)
        
        sgn = np.power(-1, inverse)
        x = x + sgn*self.circular_shift        
        x = x - self.box_length*torch.round(x/self.box_length)
        
        x = x.view(-1, self.n_particles*self.dimensions)

        part_log_det = x.new_zeros(x.shape[0], 1)

        return x, part_log_det
