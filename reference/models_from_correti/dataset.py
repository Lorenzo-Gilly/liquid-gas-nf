import numpy as np
import torch
from torch.utils.data import Dataset

from tools.util import octahedral_transformation

class PBCDataset(Dataset):
    
    def __init__(self, flow, data_tensor, test_fraction, beta_source, beta_target, shuffle_data=False, transform=True, augment=True, energy_labels=None):

        self.device = flow.device

        flow.eval()

        n_test = int(test_fraction * data_tensor.shape[0])
        random_generator = torch.Generator()

        if shuffle_data:
            shuffle_indices  = torch.randperm(data_tensor.shape[0], generator=random_generator).to(self.device)
            data_tensor      = data_tensor[shuffle_indices]
            if energy_labels is not None:
                energy_labels = energy_labels[shuffle_indices]

        assert flow.posterior.PBC, "PBCDataset can only be used for dataset with PBC"
        self.data_dimensions  = flow.posterior.dimensions
        self.data_n_particles = flow.posterior.n_particles
        self.data_box_length  = flow.posterior.box_length

        self.beta_target  = beta_target
        self.train_data_x = data_tensor[:-n_test]
        self.test_data_x  = data_tensor[-n_test:]
        if energy_labels is None:
            self.energy_train_x = flow.posterior.energy(self.train_data_x)
            self.energy_test_x  = flow.posterior.energy(self.test_data_x)
        else:
            self.energy_train_x = energy_labels[:-n_test]
            self.energy_test_x  = energy_labels[-n_test:]

        if augment:
            assert transform, "Cannot augment without transforming"

        self.augment       = augment
        self.transform     = transform

        self.beta_source   = beta_source
        self.test_data_z   = flow.prior.sample(n_test, beta=beta_source)
        self.energy_test_z = flow.prior.energy(self.test_data_z)
            

    def __len__(self):

        return self.train_data_x.shape[0]


    def __getitem__(self, idx):

        if self.augment:
            # center random particle
            train_item_p = (self.train_data_x[idx].clone()).view(self.data_n_particles, self.data_dimensions)
            indx = np.random.randint(self.data_n_particles)
            c = train_item_p[indx].clone()
            train_item_p -= c
            train_item_p -= self.data_box_length*torch.round(train_item_p/self.data_box_length)
            train_item_p[[indx, 0]] = train_item_p[[0, indx]]

            # octahedral transformations
            base_oct = octahedral_transformation(self.data_dimensions, self.device)
            full_oct = base_oct.repeat(self.data_n_particles, 1, 1)
            train_item = torch.bmm(train_item_p.unsqueeze(1), full_oct).reshape(self.data_n_particles*self.data_dimensions)
        elif self.transform:
            # center first particle
            train_item_p = (self.train_data_x[idx].clone()).view(self.data_n_particles, self.data_dimensions)
            c = train_item_p[0].clone()
            train_item_p -= c
            train_item_p -= self.data_box_length*torch.round(train_item_p/self.data_box_length)
            train_item = train_item_p.reshape(self.data_n_particles*self.data_dimensions)
        else:
            train_item = self.train_data_x[idx].clone()
        train_label = self.energy_train_x[idx]

        return train_item, train_label
    
    
    def get_test_data(self):
        
        return self.test_data_x, self.energy_test_x, self.test_data_z, self.energy_test_z