# brain-wide selectivity law

This repository contains code related to the paper:

> **A brain-wide statistical law governs neuronal selectivity**

The code provided here contains:
- `betabin-gated-vae`: code for exploring the functional advantage of the beta-binomial coding principle using ANNs (Fig. 4 in the paper).
- `visual-stimuli`: code for generating naturalistic patch stimuli for animal experiments.

## Representational performance on ANNs

> [!Note]
>
> To run this part, please ensure that you are under the directory `betabin-gated-vae/`.

### Prepare environment

##### 1. Create conda environment

```bash
conda create -n exptenv python=3.12
conda activate exptenv
```

##### 2. Install PyTorch

Please install PyTorch according to your CUDA version following the [official guide](https://pytorch.org/get-started/locally/).

For example, for PyTorch v2.7.1 with CUDA 11.8:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

##### 3. Install remaining dependencies

Remaining dependencies were written in `betabin-gated-vae/requirements.txt`.  They can be installed simply by:

```bash
pip install -r requirements.txt
```

### Run experiments

#### Train & evaluate across the parameter space

##### 1. Configure an experiment run

Two JSON files are used for configure a sweep session:

- **Model configuration** (example given as `configs/model.json`) specifies a customized Response-Gated-VAE structure;
- **Sweep configuration** (example given as `configs/sweep.json`) specifies the parameters of a experiment run.

You might edit the following entries in the **sweep configuration** to run an experiment on your computer:

- `out_dir`: Path to the directory to store the outputs for this experiment run.
- `model_cfg_path`: Path to the model configuration JSON file.
- `dataset_dir`:  Path to the folder where CIFAR-100 dataset will be downloaded.
- `is_control`: Boolean variable. If true, models will implement a binomial selectivity distribution with the same mean active fraction with the corresponding beta-binomial distribution.
- `range_alpha`, `range_beta`: sweeping range of the beta-binomial parameter space.
- `n_nodes`: number of subdivisions  of beta-binomial parameter `alpha` and `beta`.

##### 2. Train models

Run the following command to train the models:

```bash
python train_gated_vaes_param_sweep.py -p <path_to_sweep_config.json>
```

This script will:

- Created the experiment folder specified by the sweep configuration. Subdirectories with a three-digit name are corresponding to each of the models with alpha-beta configurations sampled on the parameter plane;
- Output `train_summary.csv` table under the experiment folder. The table contains brief train information of each models, and it looks like:

| node | alpha | beta | mean_activation | train_loss | val_recon | val_kld |
| ---- | ----- | ---- | --------------- | ---------- | --------- | ------- |
| ...  | ...   | ...  | ...             | ...        | ...       | ...     |

where `val_recon` and `val_kld` mean the reconstruction loss and K-L divergence of the final validation, respectively.

##### 3. Evaluate models

Run the following command to calculate RMSE and FID metrics for the models:

```
python eval_gated_vaes_param_sweep.py -p <path_to_sweep_config.json>
```

This script will:

- Evaluate the class-wise FID scores over the models under the experiment folder (Based on a boosted version modified from [pytorch-fid](https://github.com/mseitzer/pytorch-fid));
- Output `eval_summary.csv` table under the experiment folder. The table contains brief train information of each models, and it looks like:

| node | alpha | beta | mean_activation | rmse | fid |
| ---- | ----- | ---- | --------------- | ---- | --- |
| ...  | ...   | ...  | ...             | ...  | ... |

where `rmse` is computed pixel-wise, and `fid` is averaged over class-wise FID scores of the model.

#### Show performance

Please run Jupyter notebook `show_performance.ipynb`  for performance visualization. This notebook shows how to calculate and visualize the fidelity and generalization scores from saved sweeping experiments (provided under directory `outputs/sweeps`).

## Code for preparing naturalistic patch stimuli

### Prepare environment

1. Create conda environment (or use an existing one with python version above 3.12)

```bash
conda create -n stimenv python=3.12
conda activate stimenv
```

2. Install dependencies

```bash
pip install -r visual-stimuli/requirements.txt
```

Please refer to the [notebook](./visual-stimuli/naturalistic_patches.ipynb) for details about how we generated the naturalistic patch stimuli from [Hyperspectral natural imaging](https://zenodo.org/communities/hyperspectral-natural-imaging) data collected by Zimmermann et al.

## License

This project is licensed under the MIT License. Parts of the implementation in [fid.py](./betabin-gated-vae/src/evaluation/fid.py) are adapted from pytorch-fid (Apache License 2.0).