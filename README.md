# RankCL: Ranking-aware Contrastive Learning for Imbalanced Ordinal Classification

## Overview
RankCL is a framework for **contrastive learning with ordinal classification**, designed to handle imbalanced ordinal classification problems in tabular and image datasets.

---

## Environment Setup

This project was developed and tested with **Python 3.11.9**.  
All required packages are listed in `requirements.txt` or `environment.yml`.

We recommend using **PyTorch 2.3.1 with CUDA 12.1** to reproduce the results reliably.

### Option 1: Using Conda
```bash
conda env create -f environment.yaml
conda activate rankcl_env
```

### Option 2: Using Pip
```bash
conda create -n rankcl_env python=3.11.9
conda activate rankcl_env
pip install -r requirements.txt
```

---

## Dataset Setup

### Tabular Datasets

This project uses the tabular datasets provided by **ORCA**
([https://github.com/ayrna/orca](https://github.com/ayrna/orca)).
The datasets used in our experiments can be downloaded from the **official website**:
[https://www.uco.es/grupos/ayrna/orreview](https://www.uco.es/grupos/ayrna/orreview).

To prepare the datasets for this project, follow the steps below:
 
1. Download the archive **`datasets-orreview.zip`** from the [**official website**](https://www.uco.es/grupos/ayrna/orreview).
2. Place the downloaded file inside the `datasets/` directory.
3. Extract the archive to obtain the following structure:

```bash
datasets
└── datasets-orreview
    ├── discretized-regression
    │   ├── 10bins
    │   │   ├── abalone10
    │   │   ├── bank1-10
    │   │   ├── bank2-10
    │   │   ├── ...
    │   └── 5bins
    │       ├── abalone
    │       ├── bank1-5
    │       ├── bank2-5
    │       ├── ...
    └── ordinal-regression
        ├── automobile
        ├── balance-scale
        ├── bondrate
        ├── car
        ├── contact-lenses
        ├── ERA
        ├── ...

```
The dataset will be automatically loaded by our training scripts.

---
### Image Dataset
Image dataset in this project follows the dataset preparation procedure used in the `dlordinal` package.

Since the Adience dataset **cannot be downloaded automatically**, it must be prepared manually.


#### Adience Dataset

##### 1. Create the directory structure

```bash
datasets
└── adience
    └── folds
```
##### 2. Download the required files

Download the following files from the [official Adience dataset source](https://talhassner.github.io/home/projects/Adience/Adience-data.html) and place them accordingly:

`fold_0_data.txt` … `fold_4_data.txt` → `datasets/adience/folds/`

`aligned.tar.gz` → `datasets/adience/`

##### 3. Dataset Usage

The dataset will be automatically loaded by our training scripts using the `dlordinal` package.
No manual loading is required.

## Baseline Setup

This project uses a mix of official implementations and third-party re-implementations of baseline methods. 

!!!補一下rankcl會和deepOrdinal結合

### Deep Ordinal Methods

#### dlordinal

All deep ordinal methods in this repository, except for CLOC, are implemented based on the **dlordinal** framework. (see `requirements.txt` for the exact version).

- dlordinal GitHub repository: https://github.com/ayrna/dlordinal  

However, for SoftLabel-based methods, the original implementation of `get_binomial_soft_labels` does not correctly generalize to arbitrary numbers of classes.

To ensure correctness and full reproducibility, we provide a local re-implementation of the following components:

- `get_binomial_soft_labels`
- `BinomialCrossEntropyLoss` (functionality preserved with the dependency redirected to the local implementation)


under:

```bash
overrides/softlabeling.py
```

All other components of `dlordinal` remain unchanged.


#### CLOC

```bash
git clone https://github.com/dpitawela/CLOC.git
cd CLOC
git checkout bb59e65f40c1a51d0be058e6fc8e23dff6d9881c
```


### Conventional Baselines


#### OrdinalTree

The original authors of OrdinalTree did not publish official code.
Therefore, we adopt a widely-used third-party implementation from the following repository:

```bash
git clone https://github.com/mosh98/Ordinal_Classifier.git
cd Ordinal_Classifier
git checkout e917b0c5780811c841894b8e62bfbc14ed975a13
```

#### CLM

The CLM baseline denotes a linear cumulative link model (CLM) with a logistic link function implemented using `LogisticAT` from the `mord` library (see `requirements.txt` for the exact version).


#### Classical Machine Learning Baselines

The following classical machine learning baselines are implemented using widely-adopted open-source libraries. All implementations follow the official APIs of the corresponding libraries (see `requirements.txt` for the exact version).


- **Decision Tree**  
  Implemented using `DecisionTreeClassifier` from **scikit-learn**.

- **Random Forest**  
  Implemented using `RandomForestClassifier` from **scikit-learn**.

- **Multi-Layer Perceptron (MLP)**  
  Implemented using `MLPClassifier` from **scikit-learn**.

- **Logistic Regression**  
  Implemented using `LogisticRegression` from **scikit-learn**.

- **XGBoost**  
  Implemented using `XGBClassifier` from the **xgboost** library.

- **LightGBM**  
  Implemented using `LGBMClassifier` from the **lightgbm** library.



## Usage

This project uses **YAML configuration files** to manage datasets, model architectures, training hyperparameters, and RankCL-specific settings.
- For reproducibility, seeds are fixed in the training scripts.

All experiments in the paper can be reproduced by training the models from scratch. Training scripts automatically generate evaluation results upon completion.

### Train & Evaluate

```bash
# Train RankCL and automatically generate evaluation results
python main.py --config "configs/main/<dataset_type>/<dataset>/<method>.yaml"
```

* Replace `<dataset_type>`, `<dataset>`, and `<method>` with your choices:

  * For **image datasets** (e.g., FGNET, Adience), use `dataset_type=image`
  * For **tabular datasets**, use `dataset_type=tabular`
* Upon completion, evaluation metrics are automatically saved in `output/<dataset_type>/<dataset>/<method>/metrics.csv`. 

#### Quickstart: train on a small tabular dataset

```bash
# Quickstart: train on a small tabular dataset (LEV)
python main.py --config "configs/main/tabular/LEV/OBDECOC + RCM.yaml"

# After training, view evaluation results
cat "output/tabular/LEV/OBDECOC + RCM/metrics.csv"
```

### Config Structure

Below is a example of a config file (`configs/default/default.yaml`):

```yaml
method_name: "OBDECOC + RCM"
method_type: "RankCL"
base_method_name: "OBDECOC"
use_rankcl: true
use_reweight: false
out_dir: output
best_hyperparams_dir: configs/results_hyperparams
seed: 0
num_workers: 0

dataset:
  dataset_name: "LEV"
  dataset_type: "tabular"

model:
  model_params:
    enc_dims: [128, 64]
    feat_dims: [32, 16]

train:
  n_runs: 5
  epochs: 400
  batch_size: 128
  weight_decay: 0
  val_epoch: 1
  patience: 50
  best_metric_name: "QWK"
  val_ratio: 0.15

rankcl:
  with_correct_penalty: true
  similarity_metric: "cosine"  # squaredL2

```

For full experiment settings, refer to the `configs/` directory.

### Reproducibility

We fix all random seeds to **0** in our experiments and enable deterministic behavior in PyTorch to improve reproducibility:

```python
import yaml
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    torch.use_deterministic_algorithms(True, warn_only=True)

# Example usage:
# set_seed(0)
```


## Hyperparameter Tuning Ranges

The following table summarizes the hyperparameter search ranges used in our experiments.

<details>
<summary><strong>Click to expand Table XVI</strong></summary>

<br>

| Method | Hyperparameter | Search Range |
|--------|----------------|--------------|
| **RankCL** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| | λ | {0.001, 0.01, 0.1, 1, 10, 100, 1000} |
| **Decision Tree** | max_depth | [3, 20] |
| | min_samples_split | [2, 10] |
| | min_samples_leaf | [1, 10] |
| | criterion | {gini, entropy} |
| **Random Forest** | n_estimators | [100, 300] |
| | max_depth | [5, 20] |
| | min_samples_leaf | [1, 10] |
| | max_features | {sqrt, log2} |
| **XGBoost** | max_depth | [3, 12] |
| | learning rate (lr) | [1e-3, 0.3] (log scale) |
| | subsample | [0.6, 1.0] |
| **LightGBM** | num_leaves | [16, 128] |
| | learning rate (lr) | [1e-3, 0.3] (log scale) |
| | feature_fraction | [0.6, 1.0] |
| **MLP** | hidden_layer_sizes | (128, 64) |
| | activation | {relu, tanh} |
| | alpha | [1e-5, 1e-2] (log scale) |
| | learning_rate_init | [1e-4, 1e-2] (log scale) |
| **Logistic Regression** | C | [1e-3, 1e2] (log scale) |
| | solver | lbfgs |
| | multi_class | multinomial |
| | max_iter | 400 |
| | penalty | l2 |
| **Ordinal Tree** | classifier | {decision_tree, random_forest} |
|  (decision_tree)| max_depth | [3, 20] |
| | min_samples_split | [2, 10] |
| | min_samples_leaf | [1, 10] |
| | criterion | {gini, entropy} |
| (random_forest)| n_estimators | [100, 300] |
| | max_depth | [5, 20] |
| | min_samples_leaf | [1, 10] |
| | max_features | {sqrt, log2} |
| **CLM** | alpha | [1e-3, 1e2] (log scale) |
| **OBDECOC** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| **Deep CLM** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| | link_function | {logit, probit, cloglog} |
| **Deep CLMWK** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| | link_function | {logit, probit, cloglog} |
| **SoftLabel** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| | distribution | {triangular, beta, binomial, exponential} |
| | alpha2 (triangular) | {0.01, 0.05, 0.10} |
| | eta (non-triangular) | {0.8, 1.0} |
| | p (exponential) | {1.0, 1.5, 2.0} |
| **StickBreaking** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |
| **CLOR** | learning rate (lr) | {1e-4, 1e-3, 1e-2} |

</details>



## Acknowledgements and Implementation Notes

To enable integration with the proposed two-head structure in our RankCL framework, we implemented a customized **OBDECOCHead** class adapted from the open-source library **dlordinal**, specifically based on its **OBDECOCModel and ECOCOutputTransformer**.

- dlordinal GitHub repository: https://github.com/ayrna/dlordinal  
- Original authors:  
  Bérchez-Moreno, F.; Ayllón-Gavilán, R.; Vargas, V. M.; Guijo-Rubio, D.;  
  Hervás-Martínez, C.; Fernández, J. C.; Gutiérrez, P. A.  
- Copyright (c) 2024, The dlordinal developers.

This project fully follows the **BSD 3-Clause License** of the dlordinal project. We strictly respect and comply with all original license terms. All adapted components are properly credited. Any modifications are solely the responsibility of the authors of this repository.

#### Modifications
- The `forward` method has been modified to return **two-head outputs** for compatibility with the RankCL framework.
