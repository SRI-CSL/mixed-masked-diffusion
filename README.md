# MixeDiT and MaskeDiT

![Rendered topology changes while conditioning on a range of drag coefficients.](samples.png)

### Accompanying code to the paper: ``Do Diffusion Models Dream of Electric Planes?'' Discrete and Continuous Simulation-Based Inference for Aircraft Design

We generate conceptual engineering designs of electric vertical take-off and landing (eVTOL) aircraft. We follow the paradigm of simulation-based inference (SBI), whereby we look to learn a posterior distribution over the full eVTOL design space. To learn this distribution, we sample over discrete aircraft configurations (topologies) and their corresponding set of continuous parameters. Therefore, we introduce a hierarchical probabilistic model consisting of two diffusion models. The first model leverages recent work on Riemannian Diffusion Language Modeling (RDLM) and Unified World Models (UWMs) to enable us to sample topologies from a discrete and continuous space. For the second model we introduce a masked diffusion approach to sample the corresponding parameters conditioned on the topology.

**Paper**: !!!

**Dataset**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18763654.svg)](https://doi.org/10.5281/zenodo.18763654)

### Repository Structure

```
/main/

    run_maskedit_example.sh: example MaskeDiT-only conditional sampling, shell script
    run_mixed_example.sh: example full architecture conditional sampling, shell script

    plot_script.py: plotting script called by examples
    sample_mixed.py: sampling script called for sampling the full architecture

    example_settings_maskedit.json: settings for the MaskeDiT sampling example
    example_settings_mixed.json: settings for the full sampling example

    /training_data/

        /maskedit/
            /data/: training data for MaskeDiT
            /model/: trained model for MaskeDiT

        /mixedit/
            /data/: tokenizing data for MixeDiT
            /model/: trained model for MixeDiT

    /maskedit/: maskedit architecture

        /architectures/
            /simformer/
                simformer_mask.py: Masked simformer implementation in NNX
                embedding.py: Gaussian Fourier embedding
            /transformer/: transformer in NNX, called by simformer implementation

        /training/
            train_loop.py: training loop implementation, including initialization and saving
            /diffusion/: diffusion modelling folder
                diffusion_mask.py: contains individual masked diffusion training steps, validation steps and sampling script (including Euler-Maruyama)
                gaussian_prob_path.py: describes a generic Gaussian SDE, used for noising and denoising in diffusion_mask.py
                losses.py: contains various losses for different variations of diffusion training, used by diffusion_mask.py

        /datasets/: contains custom dataloaders and data normalization and processing utilities
            dataloader.py: custom dataloader
            dataset.py: custom dataset class with various utilities for dealing with masking

        /utils/indices.py: utilities used in diffusion

        sample_maskedit.py: MaskeDiT sampling function
        train_maskedit.py: MaskeDiT training function - call from command line to train MaskeDiT (see advice in function)

    /mixedit/

        /model/: Diffusion transformer model

        /tokenizing/discrete_encoding.py: tokenizer construction and tokenizing scripts

        data.py: data processing and tokenization
        distribution.py: classes for different types of distribution
        hypersphere.py: contains RDLM manifold projection operations
        losses.py: different loss functions, including the mixed loss
        main.py: entrypoint for hydra+PyTorch training and sampling
        run_sample.py: sampling function called by main.py
        run_train: training loop called by main.py
        sample_mixed_investigations.py: sampling function for when sampling the full model, called by sample_mixed in the main folder
        sampling.py: sampling loop (Euler-Maruyama)
        scheduler_lib.py: library of different diffusion schedulers
        sde.py: definition of the stochastic differential equation used for noising/denoising in MixeDiT: continuous SDE and logarithm bridge
        uav_train_mixed.py: conditional training - this is the function to call from the command line to train MixeDiT
        uav_sample_mixed.py: conditional sampling for MixeDiT - this is the function to try to sample from MixeDiT only

    /metrics/
	
            eval_loop_NLE.py: evaluating metrics on the NLE methods (also trains the NLE methods).
            eval_loop.py: evaluating metrics on the MaskeDiT model.
            run_NLE.py: Training and sampling for the NLE methods, using the `sbi` package.
            run_metrics: runs the metrics over input datasets.
            mmd.py: MMD metric
            C2ST.py: C2ST metric
			sample.py: sampling function for the MaskeDiT model metrics.

```

## User guide

### Set up the environment
A YAML file is provided for your convenience in `./environment`.

* `conda env create -f environment/environment.yml`
* `pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0`
* `pip install --upgrade \
  "jax==0.6.2" "jaxlib==0.6.2" \
  "jax-cuda12-plugin==0.6.2" "jax-cuda12-pjrt==0.6.2"`
* `pip install --upgrade "nvidia-cudnn-cu12>=9.8.0"`

### Download the Data

1. Go to [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18763654.svg)](https://doi.org/10.5281/zenodo.18763654) and download the zip file `training_data.zip`.
2. Unzip the folders such that the contents are all in the same folder, e.g. called `./training_data`. The MixeDiT dataset and trained model are in the `mixedit` subfolder and the MaskeDiT datasets (both training and testing) and trained model are in the `maskedit` folder.

### Train the MixeDiT model
To train the MixeDiT model, use `uav_train_mixed.py`. The script uses 2 inputs: a NumPy file containing discrete token IDs (`tokenized_uav.npy`) and a NumPy file containing the corresponding continuous observation features (`uav_observations.npy`). Set these paths, along with training hyperparameters and model dimensions, in the `Config` class. Running the script will automatically load or resume checkpoints from the configured `work_dir`, construct the dataset and dataloaders, and begin training. During execution, the script produces two outputs: periodic epoch‑numbered checkpoints and a `checkpoint_best.pth` file containing the model, EMA weights, optimizer state, and full configuration whenever a new best validation loss is achieved.

In the current directory run the command:

`python -m mixedit.uav_train_mixed`

### Train the MaskeDiT model
`maskedit/train_maskedit.py` trains the MaskeDiT model using command‑line arguments to specify all required inputs, including dataset paths (`--project-root`), checkpoint/output directory (`--checkpoint-root`), CUDA device selection (`--cuda-visible-devices`), model architecture parameters (e.g., `--nlayers`, `--dim-id`,`--num-heads`), diffusion settings (e.g. `--sigma-min`, `T`), and training hyperparameters (e.g. `--epochs`, `--batch-size`). The script expects `train_set.csv` and `train_indices.pkl` inside the project root and constructs masked train/validation splits automatically. All outputs—including step‑numbered checkpoints (controlled by `--save-every` and `--max-to-keep`), restored model states when `--sample` is used, generated samples (`--Nsamples`, `--sample-steps`), and a run log containing configuration and sampled data—are written to the checkpoint directory defined by `--checkpoint-root` and `--name`. Running normally performs full training; adding `--sample` loads the latest checkpoint and performs sampling only.

In the current directory run the command:

`python -m maskedit.train_maskedit --name MaskeditModel_v2`

### Conditional sampling
Run the `run_maskedit_example.sh` and `run_mixed_example.sh` shell scripts in the top level of the `main` folder.
This script automates the full investigation (conditional sampling) workflow: it prepares the environment, runs either the standard MaskeDiT sampler or the full MixeDiT-MaskeDiT pipeline, saves generated samples to the output directory, identifies the most recent investigation run, and produces summary plots using `plot_script.py`. It defines configuration paths, handles GPU selection, and generates consistent visualizations of key design metrics for each investigation. The investigation run is the `example_settings_maskedit.json` or the `example_settings_mixed.json`.
The difference between the two investigation files and their settings is that `example_settings_maskedit.json` and `run_maskedit_example.sh` runs a set of conditional samplings that do not require sampling using MixeDiT (all designs have a fixed topology), whereas `example_settings_mixed.json` and `run_mixed_example.sh` runs a set of conditional samplings that require using MixeDiT (varying topology).
To sample just the MixeDiT architecture, look at `mixedit/uav_sample_mixed.py`. Run the script after pointing `workdir` and `checkpoint_path` to your trained run directory, and it will rebuild the model/SDE, generate samples with the local sampler, and save them as `generated_samples.txt`.

### Running metrics
The `metrics` folder should be used here. Run `eval_loop.py` which evaluates MMD and C2ST metrics, and compare with `eval_loop_NLE.py`.
* `cd metrics`
* `python eval_loop.py`
* `python eval_loop_NLE.py`

## What is not included?
To generate more data, run the posterior predictive checks, or create 3D renders of the data, `FORGE` is required to interface between the open source `SUAVE` package and the generative model. `FORGE` will be released as a separate package, upon which this repo will be updated accordingly.

## Link to SUAVE package
https://github.com/suavecode/SUAVE