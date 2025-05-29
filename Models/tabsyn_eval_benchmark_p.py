import numpy as np
import pandas as pd
import json
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.preprocessing import LabelEncoder

import sys
import os
from functools import partial
from typing import Callable, Dict, List, Union

# Get the absolute path to the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
models_dir = os.path.join(root_dir, 'Models')

# Add to Python path
sys.path.insert(0, root_dir)
sys.path.insert(0, models_dir)

try:
    from Models.syn_metrics import SyntheticDataMetrics
    print("Successfully imported SyntheticDataMetrics")
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"syn_metrics.py exists: {os.path.exists(os.path.join(models_dir, 'syn_metrics.py'))}")
    sys.exit(1)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default=None, help='The file path of the synthetic data')
parser.add_argument('--n_permutations', type=int, default=10, help='Number of permutations for p-value calculation')
args = parser.parse_args()


def financial_data_imputer(data, num_col_idx, cat_col_idx):
    """Credit-risk specific imputation"""
    data = data.copy()
    num_imputer = SimpleImputer(strategy='median')
    if len(num_col_idx) > 0:
        data[num_col_idx] = num_imputer.fit_transform(data[num_col_idx])
    cat_imputer = SimpleImputer(strategy='constant', fill_value='UNKNOWN')
    if len(cat_col_idx) > 0:
        data[cat_col_idx] = cat_imputer.fit_transform(data[cat_col_idx].astype(str))
    return data


def compute_utility_metrics(real_data: pd.DataFrame, 
                          syn_data: pd.DataFrame, 
                          target_col_idx: Union[int, List[int]], 
                          task_type: str = 'classification') -> Dict[str, float]:
    """Compute utility metrics between real and synthetic data"""
    if isinstance(target_col_idx, list):
        target_col_idx = target_col_idx[0]

    # Convert target_col_idx to the actual column name (which is an integer)
    target_col = real_data.columns[target_col_idx]
    
    # Prepare data - use column index to access columns since we standardized them to integers
    X_real = real_data.drop(columns=[target_col])
    y_real = real_data[target_col]
    X_syn = syn_data.drop(columns=[target_col])
    y_syn = syn_data[target_col]

    # Encode categorical columns more robustly
    for col in X_real.columns:
        if pd.api.types.is_object_dtype(X_real[col]) or pd.api.types.is_categorical_dtype(X_real[col]):
            # Combine real and synthetic data for consistent encoding
            combined = pd.concat([X_real[col], X_syn[col]], axis=0)
            
            # Create a categorical type with all possible categories
            cat_type = pd.CategoricalDtype(categories=combined.unique(), ordered=False)
            
            # Convert both real and synthetic data using the same categories
            X_real[col] = X_real[col].astype(cat_type).cat.codes
            X_syn[col] = X_syn[col].astype(cat_type).cat.codes

    # Convert all data to float to handle potential integer codes from categorical conversion
    X_real = X_real.astype(float)
    X_syn = X_syn.astype(float)
    if task_type == "regression":
        y_real = y_real.astype(float)
        y_syn = y_syn.astype(float)
    elif task_type == "binclass":  # classification
        le = LabelEncoder()
        y_real = le.fit_transform(y_real)
        y_syn = le.transform(y_syn)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    #print("task_type",task_type)
    # Train model on synthetic data and evaluate on real data
    if task_type == 'binclass':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        
        # For binary classification, we can compute AUROC and AUPRC
        if len(np.unique(y_real)) == 2:
            y_prob = model.predict_proba(X_real)[:, 1]
            metrics = {
                'accuracy': accuracy_score(y_real, y_pred),
                'AUROC': roc_auc_score(y_real, y_prob),
                'AUPRC': average_precision_score(y_real, y_prob),
                'precision': precision_score(y_real, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_real, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_real, y_pred, average='weighted', zero_division=0)
            }
        else:
            # For multiclass, skip AUROC/AUPRC or implement multiclass versions
            metrics = {
                'accuracy': accuracy_score(y_real, y_pred),
                'precision': precision_score(y_real, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_real, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_real, y_pred, average='weighted', zero_division=0)
            }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        
        from sklearn.metrics import r2_score, mean_squared_error
        metrics = {
            'r2': r2_score(y_real, y_pred),
            'mse': mean_squared_error(y_real, y_pred)
        }
    
    return metrics


def compute_statistical_metrics(real_data: pd.DataFrame, 
                              syn_data: pd.DataFrame, 
                              num_col_idx: List[int], 
                              cat_col_idx: List[int]) -> Dict[str, float]:
    """Compute statistical similarity metrics between real and synthetic data"""
    stats = {
        'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(
            real_data, syn_data, continuous_cols=num_col_idx),
        'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(
            real_data, syn_data, categorical_cols=cat_col_idx),
        'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(
            real_data, syn_data, categorical_cols=cat_col_idx, continuous_cols=num_col_idx),
        'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(
            real_data[num_col_idx], syn_data[num_col_idx], num_col_idx),
        'JSD': SyntheticDataMetrics.calculate_jsd(
            real_data[cat_col_idx], syn_data[cat_col_idx], cat_col_idx)
    }
    return stats


def compute_privacy_metrics(real_data: pd.DataFrame, 
                          syn_data: pd.DataFrame, 
                          target_col_idx: Union[int, List[int]]) -> Dict[str, float]:
    """Compute privacy metrics between real and synthetic data"""
    target_col = target_col_idx[0] if isinstance(target_col_idx, list) else target_col_idx
    
    mi_metrics = SyntheticDataMetrics.model_inversion_attack(real_data, syn_data, target_col)
    # meminf_metrics = SyntheticDataMetrics.membership_inference_attack(real_data, syn_data, target_col)
    
    return {**mi_metrics}


def compute_p_values(real_data: pd.DataFrame,
                   syn_data: pd.DataFrame,
                   target_col_idx: Union[int, List[int]],
                   num_col_idx: List[int],
                   cat_col_idx: List[int],
                   task_type: str = 'classification',
                   n_permutations: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Compute p-values for all metrics (statistical, utility, and privacy)
    """
    p_values = {
        'statistical': {},
        'utility': {},
        'privacy': {}
    }
    
    # Compute p-values for statistical metrics
    stats_metrics = compute_statistical_metrics(real_data, syn_data, num_col_idx, cat_col_idx)
    for metric_name in stats_metrics.keys():
        # Create partial function with fixed arguments
        print("Computing for metric_name",metric_name)
        # metric_func = partial(
        #     compute_statistical_metrics,
        #     num_col_idx=num_col_idx,
        #     cat_col_idx=cat_col_idx
        # )
        p_val = permutation_test(
            compute_statistical_metrics,
            real_data,
            syn_data,
            n_permutations=n_permutations,
            metric_name=metric_name,
            num_col_idx=num_col_idx,
            cat_col_idx=cat_col_idx
        )

        p_values['statistical'][metric_name] = p_val
    
    # Compute p-values for utility metrics
    utility_metrics = compute_utility_metrics(real_data, syn_data, target_col_idx, task_type)
    for metric_name in utility_metrics.keys():
        # metric_func = partial(
        #     compute_utility_metrics,
        #     task_type=task_type
        # )
        print("Computing for metric_name",metric_name)
        p_val = permutation_test(
            compute_utility_metrics,
            real_data,
            syn_data,
            n_permutations=n_permutations,
            metric_name=metric_name,
            target_col_idx=target_col_idx,
            task_type=task_type
        )

        p_values['utility'][metric_name] = p_val
    
    # Compute p-values for privacy metrics
    privacy_metrics = compute_privacy_metrics(real_data, syn_data, target_col_idx)
    for metric_name in privacy_metrics.keys():
        print("Computing for metric_name",metric_name)
        p_val = permutation_test(
            compute_privacy_metrics,
            real_data,
            syn_data,
            n_permutations=n_permutations,
            metric_name=metric_name,
            target_col_idx=target_col_idx
        )

        p_values['privacy'][metric_name] = p_val
    
    return p_values


def permutation_test(
    metric_func: Callable, 
    real_data: pd.DataFrame, 
    syn_data: pd.DataFrame, 
    n_permutations: int = 100,
    metric_name: str = None,
    **metric_kwargs
) -> float:
    """
    Perform permutation test to calculate p-value for a given metric.
    """
    # Combine real and synthetic data
    combined_data = pd.concat([real_data, syn_data], axis=0).reset_index(drop=True)
    n_real = len(real_data)
    
    # Compute observed metric value
    observed = metric_func(real_data, syn_data, **metric_kwargs)
    if isinstance(observed, dict):
        observed = observed.get(metric_name, np.nan)

    count_extreme = 0
    
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(combined_data))
        perm_real = combined_data.iloc[perm_idx[:n_real]]
        perm_syn = combined_data.iloc[perm_idx[n_real:]]
        
        perm_value = metric_func(perm_real, perm_syn, **metric_kwargs)
        if isinstance(perm_value, dict):
            perm_value = perm_value.get(metric_name, np.nan)
        
        if abs(perm_value) >= abs(observed):
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)

def printmetrics(metrics,title=None) :
    # Print the statistical results
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == '__main__':
    #dataname = args.dataname
    model = args.model
    n_permutations = args.n_permutations
    #n_runs = args.n_runs

    for dataname in ["credit_risk_dataset","adult"]:
        print(f"\n=== Dataset: {dataname.upper()} ===")
        if not args.path:
            syn_path = f'synthetic/{dataname}/{model}.csv'
        else:
            syn_path = args.path
        real_path = f'synthetic/{dataname}/real.csv'
        data_dir = f'data/{dataname}'

        with open(f'{data_dir}/info.json', 'r') as f:
            info = json.load(f)

        real_data = pd.read_csv(real_path)
        real_data.columns = range(len(real_data.columns))

        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        target_col_idx = info['target_col_idx']

        if info['task_type'] == 'regression':
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        real_data = financial_data_imputer(real_data, num_col_idx, cat_col_idx)

        from collections import defaultdict

        stat_scores, util_scores, priv_scores, pval_scores = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(dict)
        n_runs = 10
        import subprocess
        for run in range(n_runs):
            print(f"\n======= Run {run+1}/{n_runs} =======")

            # Optional: vary dataset folder or model output path by run ID
            # cmd_vae = ["python", "main.py", "--dataname", dataname, "--method", "vae", "--mode", "train"]
            cmd_tabsyn = ["python", "main.py", "--dataname", dataname, "--method", "tabsyn", "--mode", "sample","--save_path",syn_path]

            # subprocess.run(cmd_vae, check=True)
            subprocess.run(cmd_tabsyn, check=True)

            syn_data = pd.read_csv(syn_path)
            syn_data.columns = range(len(syn_data.columns))
            syn_data = financial_data_imputer(syn_data, num_col_idx, cat_col_idx)

            stats = compute_statistical_metrics(real_data, syn_data, num_col_idx, cat_col_idx)
            printmetrics(stats,"Stats")
            utils = compute_utility_metrics(real_data, syn_data, target_col_idx, info['task_type'])
            printmetrics(utils,"Utility")
            privs = compute_privacy_metrics(real_data, syn_data, target_col_idx)
            printmetrics(privs,"Privacy")

            for k, v in stats.items():
                stat_scores[k].append(v)
            for k, v in utils.items():
                util_scores[k].append(v)
            for k, v in privs.items():
                priv_scores[k].append(v)
        print(f"\n======= Computing P values=======")
        pval_scores = compute_p_values(
            real_data, syn_data, target_col_idx, num_col_idx, cat_col_idx,
            task_type=info['task_type'], n_permutations=n_permutations
        )

        def summarize(metric_dict):
            return {k: (np.mean(v), np.std(v)) for k, v in metric_dict.items()}

        stat_summary = summarize(stat_scores)
        util_summary = summarize(util_scores)
        priv_summary = summarize(priv_scores)

        print('\n=========== Summary with Error Bars ===========')

        print("\n--- Statistical Metrics ---")
        for k, (mean, std) in stat_summary.items():
            print(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['statistical'].get(k, np.nan):.4f})")

        print("\n--- Utility Metrics ---")
        for k, (mean, std) in util_summary.items():
            print(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['utility'].get(k, np.nan):.4f})")

        print("\n--- Privacy Metrics ---")
        for k, (mean, std) in priv_summary.items():
            print(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['privacy'].get(k, np.nan):.4f})")

        # Save results
        save_dir = f'eval/quality/{dataname}'
        os.makedirs(save_dir, exist_ok=True)

        with open(f'{save_dir}/{model}.txt', 'w') as f:
            f.write(f"\n=== Statistical Metrics (mean ± std) ===\n")
            for k, (mean, std) in stat_summary.items():
                f.write(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['statistical'].get(k, np.nan):.4f})\n")

            f.write(f"\n=== Utility Metrics (mean ± std) ===\n")
            for k, (mean, std) in util_summary.items():
                f.write(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['utility'].get(k, np.nan):.4f})\n")

            f.write(f"\n=== Privacy Metrics (mean ± std) ===\n")
            for k, (mean, std) in priv_summary.items():
                f.write(f"{k}: {mean:.4f} ± {std:.4f} (p={pval_scores['privacy'].get(k, np.nan):.4f})\n")

        print(f"\nResults saved to: {save_dir}/{model}.txt")
