from torch import nn

class base_generator(nn.Module):

    def __init__(self, prior, posterior, device):
        super(base_generator, self).__init__()

        self.device = device

        self.prior = prior
        self.posterior = posterior
    

    def F_xz(self, x):
        raise NotImplementedError
    

    def F_zx(self, z):
        raise NotImplementedError
    

    def loss_xz(self, x, beta_source, beta_target):
        raise NotImplementedError
    
    
    def loss_zx(self, z, beta_target, beta_source):
        raise NotImplementedError