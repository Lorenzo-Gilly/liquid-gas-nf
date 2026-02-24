import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from tools.util import ress  # Utility function for computing effective sample size.

class Trainer(object):
    """
    A class for training a neural network using variational approaches with NLL and KLD losses.

    Attributes:
    - network: The neural network model to be trained.
    - device: The device (CPU/GPU) where computations are performed.
    """

    def __init__(self, network):
        """
        Initialize the Trainer with a network.

        Parameters:
        - network: A PyTorch model, which must define methods like loss_xz, loss_zx, and prior sampling.
        """
        self.network = network
        self.device = network.device


    def training_routine(self, train_dataset, beta_source, beta_target, w_xz=1, w_zx=1, n_epochs=50, batch_size=128,
                         n_dump=1, n_save=1, save_dir="./", optimizer=None, scheduler=None):
        """
        Perform training using the given dataset and hyperparameters.

        Parameters:
        - train_dataset: Dataset for training and validation.
        - beta_source: Source temperature (1/kT).
        - beta_target: Target temperature (1/kT).
        - w_xz: Weight for x->z loss.
        - w_zx: Weight for z->x loss.
        - n_epochs: Number of epochs for training.
        - batch_size: Number of samples per batch.
        - n_dump: Frequency of validation/logging during training (every n_dump epochs).
        - n_save: Frequency of saving model parameters (every n_save epochs).
        - save_dir: Directory for saving model parameters and logs.
        - optimizer: Optional PyTorch optimizer; uses Adam if None.
        - scheduler: Optional learning rate scheduler.
        """
        # Initialize optimizer if not provided.
        if optimizer is None:
            optimizer = torch.optim.Adam([p for p in self.network.parameters() if p.requires_grad], lr=1e-4)

        # Prepare data loader for the training dataset.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Initialize placeholders for loss metrics.
        nll = kld = torch.tensor(0., requires_grad=True).to(self.device) 
        epoch_metrics = []  # Track metrics over epochs.

        try:
            for epoch in range(1, n_epochs + 1):    
                self.network.train()  # Set the model to training mode.

                # Iterate over batches in the training data loader.
                for i, sample in enumerate(train_dataloader):
                    x = sample[0]  # Extract input data (x).

                    # Compute negative log-likelihood (x -> z) loss if enabled.
                    if w_xz > 0:
                        nll, _ = self.network.loss_xz(x, beta_source, beta_target)

                    # Compute Kullback-Leibler divergence (z -> x) loss if enabled.
                    if w_zx > 0:
                        # Sample latent configurations (z) from the prior.
                        z = self.network.prior.sample(x.shape[0], beta=beta_source)           
                        kld, _ = self.network.loss_zx(z, beta_target, beta_source)

                    # Compute total weighted loss.
                    loss_total = w_xz * nll + w_zx * kld
                    if torch.isnan(loss_total):  # Handle NaN issues during training.
                        print("\nNaN encountered in computing loss:")
                        print(f"\tnegative log-likelihood: {torch.isnan(nll)}")
                        print(f"\tKullback-Liebler divergence: {torch.isnan(kld)}")
                        raise Exception("NaN encountered in computing loss")

                    # Backpropagation and optimization step.
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

                    # Update scheduler if provided.
                    if scheduler is not None:
                        scheduler.step()

                # Validation and logging every n_dump epochs.
                if n_dump > 0:
                    if epoch == 1 or epoch % n_dump == 0:
                        self.network.eval()  # Set the model to evaluation mode.
                        
                        val_nll, val_kld, val_ress_xz, val_ress_zx = [], [], [], []
                        with torch.no_grad():
                            # Retrieve test data from the dataset.
                            test_data_x, energy_test_x, test_data_z, energy_test_z = train_dataset.get_test_data()
                            n_batches_test_data_x = max(test_data_x.shape[0] // batch_size, 1)
                            n_batches_test_data_z = max(test_data_z.shape[0] // batch_size, 1)

                            # Validate x -> z loss.
                            for b in range(n_batches_test_data_x):
                                val_nll_part, val_logw_xz_part = self.network.loss_xz(
                                    test_data_x[b * batch_size:(b + 1) * batch_size],
                                    beta_source, beta_target,
                                    energy_x=energy_test_x[b * batch_size:(b + 1) * batch_size]
                                )
                                val_ress_xz_part = ress(val_logw_xz_part)
                                val_nll.append(val_nll_part)
                                val_ress_xz.append(val_ress_xz_part)
                        
                            val_nll = torch.vstack(val_nll).mean()
                            val_ress_xz = torch.vstack(val_ress_xz).mean()

                            # Validate z -> x loss.
                            for b in range(n_batches_test_data_z):
                                val_kld_part, val_logw_zx_part = self.network.loss_zx(
                                    test_data_z[b * batch_size:(b + 1) * batch_size],
                                    beta_target, beta_source,
                                    energy_z=energy_test_z[b * batch_size:(b + 1) * batch_size]
                                )
                                val_ress_zx_part = ress(val_logw_zx_part)
                                val_kld.append(val_kld_part)
                                val_ress_zx.append(val_ress_zx_part)
                            
                            val_kld = torch.vstack(val_kld).mean()
                            val_ress_zx = torch.vstack(val_ress_zx).mean()
                            
                            val_loss_total = val_nll + val_kld

                            # Log training and validation metrics.
                            train_msg = f"epoch {epoch} | train: xz (NLL) = {nll:.3f} zx (KLD) = {kld:.3f} loss = {loss_total:.3f}"
                            train_msg += f" | eval: xz (NLL) = {val_nll:.3f} zx (KLD) = {val_kld:.3f} loss = {val_loss_total:.3f} ress_xz = {val_ress_xz:.3f} ress_zx = {val_ress_zx:.3f}"
                            if scheduler is not None:
                                train_msg += f" | lr = {scheduler.get_last_lr()[0]:.3g}"
                            print(train_msg)

                            # Store epoch metrics.
                            losses = [epoch, nll.item(), kld.item(), val_nll.item(), val_ress_xz.item(), val_kld.item(), val_ress_zx.item()]
                            if scheduler is not None:
                                losses.append(scheduler.get_last_lr()[0])
                            epoch_metrics.append(losses)

                # Save model parameters every n_save epochs.
                if n_save > 0 and epoch % n_save == 0 and epoch != n_epochs:
                    torch.save(self.network.state_dict(), os.path.join(save_dir, f"flow_parameters_{epoch}.pt"))
                    self._log_metrics(save_dir, epoch_metrics, scheduler)

            # Save final model parameters after training.
            torch.save(self.network.state_dict(), os.path.join(save_dir, f"flow_parameters.pt"))
            self._log_metrics(save_dir, epoch_metrics, scheduler)

        except KeyboardInterrupt:
            # Gracefully handle early stopping with Ctrl+C.
            return np.array(epoch_metrics)

        return np.array(epoch_metrics)


    def _log_metrics(self, save_dir, epoch_metrics, scheduler):
        """
        Helper function to log metrics to a file.

        Parameters:
        - save_dir: Directory to save the log file.
        - epoch_metrics: List of metrics collected during training.
        - scheduler: Learning rate scheduler (optional).
        """
        with open(os.path.join(save_dir, "train_log.txt"), "w+") as log_file:
            header = "# epoch\tloss_xz\tloss_zx\tval_loss_xz\tress_xz\tval_loss_zx\tress_zx"
            if scheduler is not None:
                header += "\tLR"
            log_file.write(header + "\n")
            for row in epoch_metrics:
                log_file.write("\t".join(map(str, row)) + "\n")
