# %%
import numpy as np
import matplotlib.pyplot as plt
import os

import torch

# %%
import sys
import os
from pathlib import Path

# Ensure we're in the WCA2LJ directory for correct relative paths
# This handles running from different working directories
notebook_dir = Path(__file__).parent.resolve() if '__file__' in dir() else None
if notebook_dir is None:
    # In Jupyter, find WCA2LJ directory
    cwd = Path.cwd()
    if cwd.name == 'WCA2LJ':
        notebook_dir = cwd
    elif (cwd / 'WCA2LJ').exists():
        notebook_dir = cwd / 'WCA2LJ'
    else:
        # Search for WCA2LJ in parent directories
        for parent in cwd.parents:
            if (parent / 'WCA2LJ').exists():
                notebook_dir = parent / 'WCA2LJ'
                break

if notebook_dir and Path.cwd() != notebook_dir:
    os.chdir(notebook_dir)
    print(f"Changed working directory to: {notebook_dir}")

# Add src directory to Python path
src_path = (Path.cwd().parent / 'src').resolve()
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"Added to path: {src_path}")
elif not src_path.exists():
    raise RuntimeError(f"src directory not found at {src_path}")

# %%
from systems.LJ import lennard_jones
from systems.dynamic_prior import dynamic_prior

from samplers.metropolis_MC import metropolis_monte_carlo

# %%
device = torch.device("cuda:0")

dimensions = 2
n_particles = 32
cutin = 0.8

T_source = 2
beta_source = 1/T_source
box_length_source = 6.6
rho_source = n_particles/(box_length_source)**(dimensions)
WCA = lennard_jones(n_particles=n_particles, dimensions=dimensions, rho=rho_source, device=device, cutin=cutin, cutoff="wca")
box_length_pr = WCA.box_length

T_target = 1
beta_target = 1/T_target
box_length_target = 6.6
rho_target = n_particles/(box_length_target)**(dimensions)
# rho_target = 0.70408163
# T_target = 0.60816327
# beta_target = 1/T_target
scale = (rho_source/rho_target)**(1/dimensions)
LJ = lennard_jones(n_particles=n_particles, dimensions=dimensions, rho=rho_target, device=device, cutin=cutin)
box_length_sys = LJ.box_length
# box_length_target = box_length_sys[0].item()
print(f"rho_source = {rho_source}, T_source = {T_source}")
print(f"rho_target = {rho_target}, T_target = {T_target}")
print(f"s = {scale}")

# %%
run_id = f"NVT_N{n_particles:03d}_WCA2LJ_rho_{rho_source:.2g}_T{T_source:.2g}_to_rho_{rho_target:.2g}_T{T_target:.2g}_main"
output_dir = os.path.join("./output", run_id)
assert os.path.exists(output_dir), "Folder with training parameters not found!"
gendir = os.path.join(output_dir, "generated_confs")
if not os.path.exists(gendir):
    os.makedirs(gendir)

# %%
MCMC_pr = metropolis_monte_carlo(system=WCA, step_size=0.2, n_equilibration=5000, n_cycles=1000, transform=True)
MCMC_sy = metropolis_monte_carlo(system=LJ, step_size=0.2, n_equilibration=5000, n_cycles=1000, transform=True)

# %%
wca_train_filepath = f"./data/N{WCA.n_particles:03d}/{WCA.name}/rho_{rho_source:.02g}_T_{T_source:.02g}_train.pt"
wca_sample_filepath = f"./data/N{WCA.n_particles:03d}/{WCA.name}/rho_{rho_source:.02g}_T_{T_source:.02g}_sample.pt"

print()
print("Loading WCA Training Datasets")
wca_train = torch.load(wca_train_filepath, map_location=device)
print(f"WCA Train Dataset: {wca_train_filepath}")
wca_sample = torch.load(wca_sample_filepath, map_location=device)
print(f"WCA Sample Dataset: {wca_sample_filepath}")

lj_train_filepath = f"./data/N{LJ.n_particles:03d}/{LJ.name}/rho_{rho_target:.02g}_T_{T_target:.02g}_train.pt"
lj_sample_filepath = f"./data/N{LJ.n_particles:03d}/{LJ.name}/rho_{rho_target:.02g}_T_{T_target:.02g}_sample.pt"

print()
print("Loading LJ Training Datasets")
lj_train = torch.load(lj_train_filepath, map_location=device)
print(f"LJ Train Dataset: {lj_train_filepath}")
lj_sample = torch.load(lj_sample_filepath, map_location=device)
print(f"LJ Sample Dataset: {lj_sample_filepath}")

wca_train_cpu = wca_train.view(-1, n_particles, dimensions).cpu().numpy()
wca_sample_cpu = wca_sample.view(-1, n_particles, dimensions).cpu().numpy()
lj_train_cpu = lj_train.view(-1, n_particles, dimensions).cpu().numpy()
lj_sample_cpu = lj_sample.view(-1, n_particles, dimensions).cpu().numpy()

wca_energy_train_cpu = WCA.energy(wca_train).squeeze().cpu().numpy()
lj_energy_train_cpu = LJ.energy(lj_train).squeeze().cpu().numpy()
wca_energy_sample_cpu = WCA.energy(wca_sample).squeeze().cpu().numpy()
lj_energy_sample_cpu = LJ.energy(lj_sample).squeeze().cpu().numpy()

print()
print(f"Prior train size: {wca_train.shape[0]}")
print(f"Prior sample size: {wca_sample.shape[0]}")
print(f"Posterior train size: {lj_train.shape[0]}")
print(f"Posterior sample size: {lj_sample.shape[0]}")

# %%
fig_size = (10 * 0.393701,  10 * 0.393701)
fig, ax = plt.subplots(1, 1, figsize = fig_size, dpi = 100)

ax.scatter(wca_train_cpu[::50,:,0], wca_train_cpu[::50,:,1], alpha=0.005)
ax.scatter(lj_train_cpu[::50,:,0], lj_train_cpu[::50,:,1], alpha=0.005)

plt.show()

# %%
fig_size = (10 * 0.393701,  7.5 * 0.393701)
fig, ax = plt.subplots(1, 1, figsize = fig_size, dpi = 100)

ax.hist(wca_energy_train_cpu[::10], bins=40, density=True, alpha=0.5, label="Reference WCA data")
ax.hist(wca_energy_sample_cpu[::10], bins=40, density=True, alpha=0.5, label="Reference WCA data")
ax.hist(lj_energy_train_cpu[::10], bins=40, density=True, alpha=0.5, label="Reference LJ data")
ax.hist(lj_energy_sample_cpu[::10], bins=40, density=True, alpha=0.5, label="Reference LJ data")
# ax.hist(LJ.energy(wca_train[::10]).cpu().numpy(), bins=40, density=True, alpha=0.5, label="Identity WCA to LJ")
# ax.hist(LJ.energy(wca_sample[::10]).cpu().numpy(), bins=40, density=True, label="Identity WCA to LJ")
# ax.hist(WCA.energy(lj_train[::10]).cpu().numpy(), bins=40, density=True, alpha=0.5, label="Identity LJ to WCA")
# ax.hist(WCA.energy(lj_sample[::10]).cpu().numpy(), bins=40, density=True, label="Identity LJ to WCA")

plt.show()

# %%
from tools.observables import rdf

n_bins = 100
cutoff_pr = box_length_source/2
cutoff_sys = box_length_target/2
RDF_r, RDF_wca_train = rdf(wca_train, n_particles=n_particles, dimensions=dimensions, box_length=box_length_pr, cutoff=cutoff_pr, n_bins=n_bins, batch_size=None)
RDF_r, RDF_lj_train = rdf(lj_train, n_particles=n_particles, dimensions=dimensions, box_length=box_length_sys, cutoff=cutoff_sys, n_bins=n_bins, batch_size=None)
RDF_r, RDF_wca_sample = rdf(wca_sample, n_particles=n_particles, dimensions=dimensions, box_length=box_length_pr, cutoff=cutoff_pr, n_bins=n_bins, batch_size=None)
RDF_r, RDF_lj_sample = rdf(lj_sample, n_particles=n_particles, dimensions=dimensions, box_length=box_length_sys, cutoff=cutoff_sys, n_bins=n_bins, batch_size=None)

# %%
fig_size = (10 * 0.393701,  7.5 * 0.393701)
fig, ax = plt.subplots(1, 1, figsize = fig_size, dpi = 100)

plt.plot(RDF_r, RDF_wca_train, label=r"WCA train")
plt.plot(RDF_r, RDF_wca_sample, label=r"WCA sample")
plt.plot(RDF_r, RDF_lj_train, label=r"LJ train")
plt.plot(RDF_r, RDF_lj_sample, label=r"LJ sample")
plt.legend(frameon=False)
plt.show()

# %%
WCA = dynamic_prior(n_cached=90000, test_fraction=0.1, system=WCA, sampler=MCMC_pr, init_confs=wca_train)

# %%
from normalizing_flow.equivariant_transformer import RQS_coupling_block
from normalizing_flow.circular_shift import circular_shift

n_bins = 16

block_list = [
    
    # Block 1
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # Block 2
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # Block 3
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # Block 4
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # Block 5
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # Block 6
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # Block 7
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # Block 8
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    circular_shift(n_particles-1, dimensions, device),
    RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # # Block 9
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # # Block 10
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # # Block 11
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # # Block 12
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # # Block 13
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # # Block 14
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    # # Block 15
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    
    # # Block 16
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    
    # circular_shift(n_particles-1, dimensions, device),
    # RQS_coupling_block((1,), n_particles-1, dimensions, device, n_bins),
    # RQS_coupling_block((0,), n_particles-1, dimensions, device, n_bins),

    ]

# %%
from transformations.normalization import normalize_box
from transformations.remove_origin import remove_origin

norm_box_pr = normalize_box(n_particles=n_particles, dimensions=dimensions, box_length=box_length_pr, device=device)
norm_box_sys = normalize_box(n_particles=n_particles, dimensions=dimensions, box_length=box_length_sys, device=device)

rm_origin = remove_origin(n_particles=n_particles, dimensions=dimensions, device=device)

# %%
from normalizing_flow.flow_assembler import flow_assembler

flow = flow_assembler(prior = WCA, posterior = LJ, device=device, 
                        blocks = block_list,
                        prior_sided_transformation_layers = [norm_box_pr, rm_origin], 
                        post_sided_transformation_layers = [norm_box_sys, rm_origin]
                        ).to(device)

print(f"Flow parameters: {sum(p.numel() for p in flow.parameters() if p.requires_grad)}")

# %%
flow_parameters_filepath = os.path.join(output_dir, "flow_parameters.pt")
print(f"Loading network parameters from {flow_parameters_filepath}")
flow.load_state_dict(torch.load(flow_parameters_filepath))
metrics_filepath = os.path.join(output_dir, "train_log.txt")
print(f"Loading metrics from {metrics_filepath}")
metrics = np.loadtxt(metrics_filepath)

# %%
length_x = 50 if metrics.shape[1] <= 8 else 75
fig_size = (length_x * 0.393701, 10 * 0.393701)
fig, ax = plt.subplots(1, 3 if metrics.shape[1] <= 8 else 5, figsize = fig_size, dpi = 600)

if metrics[:,1].mean() != 0:
    ax[0].plot(metrics[:,0], metrics[:,1], label="train", color="C0")
ax[0].plot(metrics[:,0], metrics[:,3], label="eval", color="C1")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("NLL loss")

if metrics[:,2].mean() != 0:
    ax[1].plot(metrics[:,0], metrics[:,2], label="train", color="C0")
ax[1].plot(metrics[:,0], metrics[:,5], label="eval", color="C1")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("KLD loss")
ax[1].legend(frameon=False)

ax[2].plot(metrics[:,0], metrics[:,4], label=r"$x\to z$", color="C2")
ax[2].plot(metrics[:,0], metrics[:,6], label=r"$z\to x$", color="C3")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("ress")
ax[2].set_yscale("log")
ax[2].set_ylim(None,1)
ax[2].legend(frameon=False)

plt.show()

# %%
from tqdm import tqdm
import datetime
from tools.util import ress

load_replicas_zx = False

N_replicas = 10
NN_generated_samples = 500000
NN_replica_size = NN_generated_samples//N_replicas

print(f"NN_generated_samples = {NN_generated_samples}")
print(f"N_replicas = {N_replicas}")
print(f"NN_replica_size = {NN_replica_size}")
print()

if load_replicas_zx:
    print("Loading source configurations from file")
else:
    print("Generating source configurations")
    print("Equilibrating sampler")
    z, _, _ = MCMC_pr.sample_space(NN_replica_size, 0.2*beta_source)
    MCMC_pr.equilibrated = False
    z, _, _ = MCMC_pr.sample_space(NN_replica_size, beta_source)

print("Transforming source to target... ", flush=True, end = "")
with open(os.path.join(gendir, "time_zx.out"), "w") as fout_t:
    fout_t.write("# replica\tsample\ttransform\n")
    with open(os.path.join(gendir, "ress_zx.out"), "w") as fout:
        fout.write("# replica\tid_ress_zx\tress_zx\n")
        for r in tqdm(range(N_replicas), ncols=100, desc="\tProgress"):
            # PyTorch does not need the gradient for the transformation 
            with torch.no_grad():

                flow.eval()

                # load replicas previously generated
                sample_stime = datetime.datetime.now()
                if load_replicas_zx:
                    z_generated_filepath = os.path.join(gendir, f"z_{r:04d}.pt")
                    z = torch.load(z_generated_filepath, map_location=device)
                else: # generate replicas
                    z, _, _ = MCMC_pr.sample_space(NN_replica_size, beta_source)
                    z_generated_filepath = os.path.join(gendir, f"z_{r:04d}.pt")
                    torch.save(z, z_generated_filepath)

                # Transforming from latent to target via the Normalizing Flow
                sample_etime = datetime.datetime.now()
                Tz, logJ_zx = flow.F_zx(z)
                transf_etime = datetime.datetime.now()

                # Compute energy of identity transformations
                id_energy_x = flow.posterior.energy(z*scale)

                # Computing weights
                id_log_prob_zx = -beta_target*id_energy_x
                id_log_prob_z = -beta_source*flow.prior.energy(z)        
                id_log_w = (id_log_prob_zx - id_log_prob_z).squeeze(-1)
                id_ress_zx = ress(id_log_w)

                # Compute energy of transformed configurations
                WCA2LJ_energy_transformed = (LJ.energy(Tz)).cpu().numpy()

                # Computing weights
                log_prob_zx = -beta_target*flow.posterior.energy(Tz)
                log_prob_z = -beta_source*flow.prior.energy(z)        
                log_w_zx = (log_prob_zx - log_prob_z + logJ_zx).squeeze(-1)
                ress_zx = ress(log_w_zx)

                # Resampling to obtain unbiased target distribution
                Tz_cpu = Tz.view(-1, n_particles, dimensions).cpu().numpy()
                w_zx = torch.exp(log_w_zx - torch.max(log_w_zx)).cpu().numpy()
                N = Tz_cpu.shape[0]
                indx = np.random.choice(np.arange(0, N), replace=True, size = N, p = w_zx/np.sum(w_zx))
                Tz_resampled = torch.from_numpy(Tz_cpu[indx].reshape(-1, n_particles*dimensions)).to(device)

                fout_t.write(f"{r}\t{sample_etime-sample_stime}\t{transf_etime-sample_etime}\n")
                fout.write(f"{r}\t{id_ress_zx}\t{ress_zx}\n")

            Tz_generated_filepath = os.path.join(gendir, f"Tz_{r:04d}.pt")
            torch.save(Tz, Tz_generated_filepath)
            log_w_zx_generated_filepath = os.path.join(gendir, f"log_w_zx_{r:04d}.pt")
            torch.save(log_w_zx, log_w_zx_generated_filepath)
            Tz_resampled_generated_filepath = os.path.join(gendir, f"Tz_resampled_{r:04d}.pt")
            torch.save(Tz_resampled, Tz_resampled_generated_filepath)

print("Done", flush=True)

# %%
from tqdm import tqdm

load_replicas_xz = False

N_replicas = 10
NN_generated_samples = 500000
NN_replica_size = NN_generated_samples//N_replicas

print(f"NN_generated_samples = {NN_generated_samples}")
print(f"N_replicas = {N_replicas}")
print(f"NN_replica_size = {NN_replica_size}")
print()

if load_replicas_xz:
    print("Loading target configurations from file")
else:
    print("Generating target configurations")
    print("Equilibrating sampler")
    x, _, _ = MCMC_sy.sample_space(NN_replica_size, 0.2*beta_target)
    MCMC_sy.equilibrated = False
    x, _, _ = MCMC_sy.sample_space(NN_replica_size, beta_target)

print("Transforming target to source... ", flush=True, end = "")
with open(os.path.join(gendir, "ress_xz.out"), "w") as fout:
    fout.write("# replica\tid_ress_xz\tress_xz\n")
    for r in tqdm(range(N_replicas), ncols=100, desc="\tProgress"):
        # PyTorch does not need the gradient for the transformation 
        with torch.no_grad():

            flow.eval()

            # load replicas previously generated
            if load_replicas_xz:
                x_generated_filepath = os.path.join(gendir, f"x_{r:04d}.pt")
                x = torch.load(x_generated_filepath, map_location=device)
            else: # generate replicas
                x, _, _ = MCMC_sy.sample_space(NN_replica_size, beta_target)
                x_generated_filepath = os.path.join(gendir, f"x_{r:04d}.pt")
                torch.save(x, x_generated_filepath)

            # Transforming from latent to target via the Normalizing Flow
            Tinvx, logJ_xz = flow.F_xz(x)

            # Compute energy of identity transformations
            id_energy_z = flow.prior.energy(x/scale)

            # Computing weights
            id_log_prob_xz = -beta_source*id_energy_z
            id_log_prob_x = -beta_target*flow.posterior.energy(x)        
            id_log_w = (id_log_prob_xz - id_log_prob_x).squeeze(-1)
            id_ress_xz = ress(id_log_w)

            # Compute energy of transformed configurations
            LJ2WCA_energy_transformed = (WCA.energy(Tinvx)).cpu().numpy()

            # Computing weights
            log_prob_xz = -beta_source*flow.prior.energy(Tinvx)
            log_prob_x = -beta_target*flow.posterior.energy(x)        
            log_w_xz = (log_prob_xz - log_prob_x + logJ_xz).squeeze(-1)
            ress_xz = ress(log_w_xz)

            fout.write(f"{r}\t{id_ress_xz}\t{ress_xz}\n")

            # Resampling to obtain unbiased target distribution
            Tinvx_cpu = Tinvx.view(-1, n_particles, dimensions).cpu().numpy()
            w_xz = torch.exp(log_w_xz - torch.max(log_w_xz)).cpu().numpy()
            N = Tinvx_cpu.shape[0]
            indx = np.random.choice(np.arange(0, N), replace=True, size = N, p = w_xz/np.sum(w_xz))
            Tinvx_resampled = torch.from_numpy(Tinvx_cpu[indx].reshape(-1, n_particles*dimensions)).to(device)
        
        Tinvx_generated_filepath = os.path.join(gendir, f"Tinvx_{r:04d}.pt")
        torch.save(Tinvx, Tinvx_generated_filepath)
        log_w_xz_generated_filepath = os.path.join(gendir, f"log_w_xz_{r:04d}.pt")
        torch.save(log_w_xz, log_w_xz_generated_filepath)
        Tinvx_resampled_generated_filepath = os.path.join(gendir, f"Tinvx_resampled_{r:04d}.pt")
        torch.save(Tinvx_resampled, Tinvx_resampled_generated_filepath)

print("Done", flush=True)

# %%



