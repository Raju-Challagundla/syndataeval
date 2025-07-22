# SyndataEval: Benchmarking Synthetic Tabular Data Generators

This repository contains the experimental framework and benchmarking code used in our comprehensive survey on synthetic tabular data generation techniques. It provides a unified evaluation pipeline for comparing popular generative models across multiple datasets using a range of utility, statistical, and privacy metrics.

## Purpose

The goal of this repository is to support reproducible evaluation of state-of-the-art tabular data generators. It enables researchers and practitioners to:
- Train and evaluate generative models on structured datasets
- Quantitatively compare models using standardized metrics
- Understand trade-offs across utility, privacy, and statistical fidelity

This benchmarking code was developed as part of our survey paper:

> **Synthetic Tabular Data Generation: A Comparative Survey for Modern Techniques**  
> [Raju Challagundla] • [University of North Carolina at Charlotte]  
> [Link to Paper](#https://www.arxiv.org/abs/2507.11590)

## Included Models

The following synthetic data generation models are supported:

- **CTGAN** — Conditional GAN for tabular data [(SDV)](https://github.com/sdv-dev/CTGAN)
- **PATEGAN** — Private Aggregation of Teacher Ensembles GAN [(smartnoise)](https://github.com/opendp/smartnoise-sdk)
- **DPCTGAN** — Differentially Private Conditional Tabular GAN [(smartnoise)](https://github.com/opendp/smartnoise-sdk)
- **FCTGAN** — Fourier Conditional Tabular GAN [(original repo)](https://github.com/ethan-keller/FCT-GAN)
- **CTAB-GAN** — A novel conditional table GAN architecture that can effectively model diverse data types, including a mix of continuous and categorical variables [(Team-TUD)](https://github.com/Team-TUD/CTAB-GAN.git)
- **Tabsyn** — Self-Attention based transformer for tabular synthetic data [(Amazon Science)](https://github.com/amazon-science/tabsyn)


## Datasets

Experiments are conducted on widely-used benchmark datasets:
- **Adult Income** (UCI)
- **Credit Risk Default** (Kaggle)
- (Extendable to additional tabular datasets)

## Evaluation Metrics

We evaluate generated synthetic data on the following dimensions:

### Utility
- Classifier performance 
- Cross-dataset generalization (Train on synthetic, test on real and vice versa)

### Statistical Fidelity
- Marginal distribution distance (e.g., KS-statistics)
- Pairwise correlations
- Feature importance overlap

### Privacy
- Membership Inference Attack (MIA)
- Attribute Inference Attack (AIA)
- Distance to Closest Record (DCR)

All metrics include support for error bars (mean ± std) and significance testing via permutation-based p-values.

## Structure

```bash
syndataeval/
│
├── Models/           # Wrappers for synthetic data generators
├── Real_Datasets/             # Preprocessing and dataset loading
├── Experiment_Results/      # Scripts to run full experiments

---

## Repository Setup

Clone the repo and initialize submodules:

git clone https://github.com/Raju-Challagundla/syndataeval
cd syndataeval

# Submodules for models
git submodule add https://github.com/Team-TUD/CTAB-GAN.git models/CTAB-GAN
git submodule add https://github.com/ethan-keller/FCT-GAN.git models/FCT-GAN-main
git submodule add https://github.com/amazon-science/tabsyn.git models/tabsyn

Install required Python packages:

```bash
# For CTGAN (SDV library)
pip install sdv

# For PATEGAN and DPCTGAN (via Microsoft SmartNoise)
pip install smartnoise-synth
---

## Run Main Benchmark

To run experiments for CTGAN, DPCTGAN, PATEGAN, CTAB-GAN, and FCT-GAN, use:

```bash
python syn_eval_benchmark_v2.py
```

This script will:
- Train models
- Generate synthetic datasets
- Compute utility, privacy, and statistical metrics
- Save results in the `Experiment_Results/` directory

> Make sure required models are downloaded via submodules beforehand.

---

## Tabsyn Workflow (Run Separately)

Tabsyn requires a different environment and execution flow.

### Environment Setup

```bash
conda activate synthcity
```

### Train Tabsyn

#### Adult Dataset

```bash
# Train VAE first
python main.py --dataname adult --method vae --mode train

# Then train Tabsyn
python main.py --dataname adult --method tabsyn --mode train
```

#### Credit Risk Dataset

```bash
# Train VAE first
python main.py --dataname credit --method vae --mode train

# Then train Tabsyn
python main.py --dataname credit --method tabsyn --mode train
```

### Evaluate Tabsyn

From within the `models/tabsyn` folder:

```bash
# Evaluate on Adult dataset
python eval/eval_quality_imputed.py --dataname adult --model tabsyn --path synthetic/adult/tabsyn.csv

# Evaluate on Credit Risk dataset
python eval/eval_quality_imputed.py --dataname credit_risk_dataset --model tabsyn --path synthetic/credit_risk_dataset/tabsyn.csv
```

---

## Output

- Summary CSVs: mean, std
- P-value tables
- Privacy-Utility plots (optional)

---
## Generating Plots

To generate comparison plots from the benchmarking results, run:

```bash
synth_plots.ipynb
```

This script visualizes:
- Utility and privacy trade-offs
- Performance comparisons across models
- Dataset-specific metric summaries

Plots will be saved in the `Experiment_Results/plots/` directory.

## Citation

If you use this codebase or the survey in your work, please cite:

```bibtex
@misc{challagundla2025synthetictabulardatageneration,
      title={Synthetic Tabular Data Generation: A Comparative Survey for Modern Techniques}, 
      author={Raju Challagundla and Mohsen Dorodchi and Pu Wang and Minwoo Lee},
      year={2025},
      eprint={2507.11590},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.11590}, 
}
```

---

## Contact

For questions, suggestions, or collaborations, feel free to open an issue or contact [rchalla5@charlotte.edu](mailto:rchalla5@charlotte.edu).