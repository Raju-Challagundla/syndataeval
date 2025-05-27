import numpy as np
import pandas as pd
import json
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
#from ..Models.syn_metrics import SyntheticDataMetrics

pd.options.mode.chained_assignment = None

import sys
import os

# Get the absolute path to the root directory (two levels up from eval.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
models_dir = os.path.join(root_dir, 'Models')

# Add to Python path
sys.path.insert(0, root_dir)
sys.path.insert(0, models_dir)

# Now import using the full package path
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

args = parser.parse_args()

def financial_data_imputer(data, num_col_idx, cat_col_idx):
    """Credit-risk specific imputation"""
    data = data.copy()
    
    # Numerical imputation (median for robustness)
    num_imputer = SimpleImputer(strategy='median')
    if len(num_col_idx) > 0:
        data[num_col_idx] = num_imputer.fit_transform(data[num_col_idx])
    
    # Categorical imputation (constant 'UNKNOWN')
    cat_imputer = SimpleImputer(strategy='constant', fill_value='UNKNOWN')
    if len(cat_col_idx) > 0:
        data[cat_col_idx] = cat_imputer.fit_transform(data[cat_col_idx].astype(str))
    
    return data

if __name__ == '__main__':
    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path
    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 

    print(f"Evaluating synthetic data from: {syn_path}")
    
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    # Load and impute data
    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    # Standardize column names
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Apply credit-risk specific imputation
    real_data = financial_data_imputer(real_data, num_col_idx, cat_col_idx)
    syn_data = financial_data_imputer(syn_data, num_col_idx, cat_col_idx)
        
    # Calculate statistical metrics using SyntheticDataMetrics
    print('\n=========== Statistical Similarity Metrics ===========')
    stats = {
        'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(real_data, syn_data, continuous_cols=num_col_idx),
        'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(real_data, syn_data, categorical_cols=cat_col_idx),
        'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(real_data, syn_data, categorical_cols=cat_col_idx, continuous_cols=num_col_idx),
        'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(real_data[num_col_idx], syn_data[num_col_idx], num_col_idx),
        'JSD': SyntheticDataMetrics.calculate_jsd(real_data[cat_col_idx], syn_data[cat_col_idx], cat_col_idx)
    }
    
    # Print the statistical results
    for metric, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Calculate privacy metrics
    print('\n=========== Privacy Metrics ===========')
    target_col = target_col_idx[0] if isinstance(target_col_idx, list) else target_col_idx
    
    # Model inversion attack metrics
    mi_metrics = SyntheticDataMetrics.model_inversion_attack(real_data, syn_data, target_col)
    print("\nModel Inversion Attack Results:")
    for k, v in mi_metrics.items():
        if v is None:
            print(f"  {k}: Not Applicable")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Membership inference attack metrics
    meminf_metrics = SyntheticDataMetrics.membership_inference_attack(real_data, syn_data, target_col)
    print("\nMembership Inference Attack Results:")
    for k, v in meminf_metrics.items():
        if v is None:
            print(f"  {k}: Not Applicable")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Combine all attack metrics
    attack_metrics = {**mi_metrics, **meminf_metrics}

    # Preprocessing pipeline
    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    # Convert to numpy arrays
    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    # Handle special cases (original logic preserved)
    if (dataname == 'default' or dataname == 'news') and model[:4] == 'codi':
        cat_syn_data_np = cat_syn_data.astype('int').to_numpy().astype('str')
    elif model[:5] == 'great':
        if dataname == 'shoppers':
            cat_syn_data_np[:, 1] = cat_syn_data[11].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 2] = cat_syn_data[12].astype('int').to_numpy().astype('str')
            cat_syn_data_np[:, 3] = cat_syn_data[13].astype('int').to_numpy().astype('str')
            
            max_data = cat_real_data[14].max()
            cat_syn_data.loc[cat_syn_data[14] > max_data, 14] = max_data
            cat_syn_data_np[:, 4] = cat_syn_data[14].astype('int').to_numpy().astype('str')
        
        elif dataname in ['default', 'faults', 'beijing']:
            columns = cat_real_data.columns
            for i, col in enumerate(columns):
                if (cat_real_data[col].dtype == 'int'):
                    max_data = cat_real_data[col].max()
                    min_data = cat_real_data[col].min()
                    cat_syn_data.loc[cat_syn_data[col] > max_data, col] = max_data
                    cat_syn_data.loc[cat_syn_data[col] < min_data, col] = min_data
                    cat_syn_data_np[:, i] = cat_syn_data[col].astype('int').to_numpy().astype('str')

    # One-hot encoding with NaN protection
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    # Prepare final datasets with imputation check
    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis=1).astype(float))
    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1).astype(float))

    # Final NaN check before evaluation
    assert not le_real_data.isna().any().any(), "Real data still contains NaNs after imputation"
    assert not le_syn_data.isna().any().any(), "Synthetic data still contains NaNs after imputation"

    # Alpha Precision evaluation
    print('\n=========== Alpha/Beta Metrics ===========')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    try:
        quality_evaluator = eval_statistical.AlphaPrecision()
        qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
        qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}
        
        print('Alpha precision: {:.6f}, Beta recall: {:.6f}'.format(
            qual_res['delta_precision_alpha_naive'], 
            qual_res['delta_coverage_beta_naive']))
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        qual_res = {'delta_precision_alpha_naive': np.nan, 
                   'delta_coverage_beta_naive': np.nan}

    # Save all results
    save_dir = f'eval/quality/{dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/{model}.txt', 'w') as f:
        # Original metrics
        f.write(f"Alpha Precision: {qual_res['delta_precision_alpha_naive']}\n")
        f.write(f"Beta Recall: {qual_res['delta_coverage_beta_naive']}\n")
        
        # Statistical metrics from SyntheticDataMetrics
        for metric, value in stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        # Privacy metrics
        f.write("\n=== Privacy Metrics ===\n")
        for metric, value in attack_metrics.items():
            if value is None:
                f.write(f"{metric}: Not Applicable\n")
            elif isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")

    print(f"\nResults saved to: {save_dir}/{model}.txt")