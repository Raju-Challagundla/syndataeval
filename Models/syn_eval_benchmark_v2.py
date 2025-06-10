import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, mean_absolute_percentage_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import torch
from tqdm import tqdm
from Models.syn_metrics import SyntheticDataMetrics
from snsynth import Synthesizer
from snsynth.pytorch.nn.pategan import PATEGAN
from snsynth.pytorch.nn.dpctgan import DPCTGAN

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Go up TWICE to project root
adult_dataset_path = os.path.join(root_dir, 'Real_Datasets', 'Adult_dataset.csv')
credit_dataset_path = os.path.join(root_dir, 'Real_Datasets', 'credit_risk_dataset.csv')
pkl_dir = os.path.join(root_dir, 'pkl')
os.makedirs(pkl_dir, exist_ok=True)

import sys
from pathlib import Path

# Add the parent directory of FCT-GAN-main to Python path
# fctgan_path = str(Path(__file__).parent.parent / "FCT-GAN-main")
# print(f"Current Python path: {sys.path}")
# print(f"Adding to path: {fctgan_path}")
# sys.path.insert(0, fctgan_path)

from model.fctgan import FCTGAN  # Should work now

# Add model directories to Python path
import sys
#fctgan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FCT-GAN-main')
ctabgan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CTAB-GAN')

# fctgan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
# #sys.path.insert(0, fctgan_parent)
# sys.path.insert(0, fctgan_path)
# print("fctgan_path",fctgan_path)
sys.path.insert(0, ctabgan_path)
#from model.synthesizer import ImageTransformer

try:
    from model.fctgan import FCTGAN
except ImportError as e:
    print(f"Warning: Could not import FCTGAN - some functionality will be unavailable. Error: {e}")
    FCTGAN = None

try:
    from model.ctabgan import CTABGAN
except ImportError as e:
    print(f"Warning: Could not import CTABGAN - some functionality will be unavailable. Error: {e}")
    CTABGAN = None

# Check for GPU
if torch.cuda.is_available():
    print("CUDA available. Running on GPU.")
else:
    print("CUDA not available. Running on CPU.")

# Define column types for each dataset
DATASET_COLUMNS = {
    "adult": {
        "continuous": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
        "categorical": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country','income'],
        "mixed": {
            'education': ['hours-per-week'],
            'occupation': ['capital-gain'],
            'marital-status': ['age']
        },
        "target": 'income'
    },
    "credit": {
        "continuous": ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'],
        "categorical": ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file','loan_status'],
        "mixed": {
            'loan_grade': ['loan_int_rate'],
            'person_home_ownership': ['person_income'],
            'loan_intent': ['loan_amnt']
        },
        "target": 'loan_status'
    }
}

def load_dataset(name):
    """Load and return dataset along with target column"""
    if name == "adult":
        data = pd.read_csv(adult_dataset_path)
    elif name == "credit":
        data = pd.read_csv(credit_dataset_path)
    else:
        raise ValueError("Unknown dataset")
    return data, DATASET_COLUMNS[name]["target"]

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    mean_absolute_percentage_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd

def train_model(X_train, y_train, X_test, y_test, classification=True):
    """Train and evaluate a model with automatic type safety and encoding"""

    # Convert to DataFrame if not already (helps with column selection)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

       # Detect and encode categorical features using OrdinalEncoder
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        # Convert to strings to prevent type mismatches and ensure consistency
        X_train[cat_cols] = X_train[cat_cols].astype(str)
        X_test[cat_cols] = X_test[cat_cols].astype(str)

        # Handle NaNs explicitly
        X_train[cat_cols] = X_train[cat_cols].fillna("NA")
        X_test[cat_cols] = X_test[cat_cols].fillna("NA")

        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = enc.transform(X_test[cat_cols])


    # Convert target to numpy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if classification:
        # Ensure classification targets are integers
        if not np.issubdtype(y_train.dtype, np.integer):
            try:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
            except (ValueError, TypeError):
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

        # Verify classification makes sense
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 100:
            raise ValueError(f"Too many classes ({len(unique_classes)}). Did you mean regression?")

        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_proba = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            y_proba = proba[:, 1] if proba.shape[1] == 2 else None

        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred, average='weighted'),
            'AUROC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'AUPRC': average_precision_score(y_test, y_proba) if y_proba is not None else None,
            'Classes': str(unique_classes)
        }

    else:
        # Regression
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        reg = RandomForestRegressor(random_state=0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        return {
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'EVS': explained_variance_score(y_test, y_pred),
            'Target_range': f"{y_train.min():.2f}-{y_train.max():.2f}"
        }


def initialize_fctgan(dataset_name, real_df):
    """Initialize FCT-GAN with appropriate parameters"""
    config = {
        "dataset": dataset_name.capitalize(),
        "raw_csv_path": adult_dataset_path if dataset_name == "adult" else credit_dataset_path,  # We're passing dataframe directly
        "test_ratio": 0.20,
        "categorical_columns": DATASET_COLUMNS[dataset_name]["categorical"],
        "log_columns": [],
        "mixed_columns": DATASET_COLUMNS[dataset_name]["mixed"],
        "integer_columns": DATASET_COLUMNS[dataset_name]["continuous"],
        "problem_type": {"Classification": DATASET_COLUMNS[dataset_name]["target"]},
        "epochs": 150
    }
   
    return FCTGAN(**config)

def initialize_ctabgan(dataset_name, real_df):
    """Initialize CTABGAN with appropriate parameters"""
    return CTABGAN(
        raw_csv_path=adult_dataset_path if dataset_name == "adult" else credit_dataset_path,  # We'll pass data directly
        test_ratio=0.20,
        categorical_columns=DATASET_COLUMNS[dataset_name]["categorical"],
        log_columns=[],
        mixed_columns={col: [] for col in DATASET_COLUMNS[dataset_name]["mixed"]},
        integer_columns=DATASET_COLUMNS[dataset_name]["continuous"],
        problem_type={"Classification": DATASET_COLUMNS[dataset_name]["target"]},
        epochs=150
    )

# def evaluate_synthetic_data_model(real_df, target, synth_size, dataset_name, model_name, model_class, model_params):
#     """
#     Evaluate synthetic data generated by CTGAN, FCT-GAN or CTABGAN
#     Returns utility metrics and attack metrics
#     """
#     # Get columns for this dataset
#     continuous = DATASET_COLUMNS[dataset_name]["continuous"]
#     categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    
#     # Prepare real data
#     X_real = real_df.drop(columns=[target])
#     y_real = real_df[target]
#     classification = SyntheticDataMetrics.is_classification(y_real)
    
#     # Split real data for evaluation
#     X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
#         X_real, y_real, test_size=0.3, random_state=42
#     )
#     print(f"\n--- Training {model_name} on {dataset_name} dataset ---")
#     # Initialize and train synthesizer
#     if model_name in ["dpctgan", "pategan"]:
#         synthesizer = Synthesizer.create(model_class, **model_params)
#         synthesizer.fit(real_df, preprocessor_eps=0.2)
#     elif model_name == "ctgan":
#         metadata = SingleTableMetadata()
#         metadata.detect_from_dataframe(real_df)
#         synthesizer = model_class(metadata, **model_params)
#         synthesizer.fit(real_df)
#     elif model_name == "fctgan" and FCTGAN is not None:
#         synthesizer = initialize_fctgan(dataset_name, real_df)
#         synthesizer.fit()
#     elif model_name == "ctabgan" and CTABGAN is not None:
#         synthesizer = initialize_ctabgan(dataset_name, real_df)
#         synthesizer.fit()
#     else:
#         raise ValueError(f"Unknown or unavailable model: {model_name}")
    
    
#     # Train synthesizer
#     # if model_name == "ctgan":
#     #     synthesizer.fit(real_df)
#     # elif model_name == "fctgan":
#     #     synthesizer.fit(real_df, None)
#     # elif model_name == "ctabgan":
#     #     synthesizer.fit(real_df)
#     print(f"\n--- Generating Samples ---")
#     # Generate synthetic data
#     if model_name in ["fctgan", "ctabgan"]:
#         synth_data = synthesizer.generate_samples()
#     else:
#         synth_data = synthesizer.sample(synth_size)

    
#     # Prepare synthetic features/target
#     if target not in synth_data.columns:
#         print(f"Warning: Target column {target} not found in synthetic data")
#         return None, None
    
#     X_synth = synth_data.drop(columns=[target])
#     y_synth = synth_data[target]
    
#     # Calculate statistical metrics
#     stats = {
#         'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(real_df, synth_data, continuous_cols=continuous),
#         'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(real_df, synth_data, categorical_cols=categorical),
#         'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(real_df, synth_data, categorical_cols=categorical, continuous_cols=continuous),
#         'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(real_df[continuous], synth_data[continuous], continuous),
#         'JSD': SyntheticDataMetrics.calculate_jsd(real_df[categorical], synth_data[categorical], categorical)
#     }
    
#     # Utility evaluation
#     utility_metrics = train_model(X_synth, y_synth, X_real_test, y_real_test, classification)
#     utility_metrics.update(stats)
    
#     print("\nStatistical Metrics:")
#     for k, v in stats.items():
#         print(f"{k}: {v:.4f}")
    
#     print("\nUtility Metrics:")
#     for k, v in utility_metrics.items():
#         if v is not None:
#             try:
#                 # Try float formatting if possible
#                 print(f"{k}: {float(v):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
#             except (ValueError, TypeError):
#                 # Fallback to regular string output
#                 print(f"{k}: {v}")
    
#     # Privacy attacks
#     mi_metrics = SyntheticDataMetrics.model_inversion_attack(real_df, synth_data, target)
#     #meminf_metrics = SyntheticDataMetrics.membership_inference_attack(real_df, synth_data, target)
#     attack_metrics = {**mi_metrics}
    
#     print("\nPrivacy Attack Results:")
#     for k, v in attack_metrics.items():
#         if v is None:
#             print(f"  {k}: Not Applicable")
#         elif isinstance(v, (int, float)):
#             print(f"  {k}: {v:.4f}")
#         else:
#             print(f"  {k}: {v}")
    
#     # Save model
#     model_path = os.path.join(pkl_dir, f"{model_name}_{dataset_name}.pkl")
#     joblib.dump(synthesizer, model_path)
#     print(f"\nModel saved to {model_path}")
    
#     return utility_metrics, attack_metrics
def evaluate_synthetic_data_model(real_df, target, synth_size, dataset_name, 
                                model_name, model_class, model_params, num_trials=5):
    """
    Evaluates synthetic data by:
    1. Training the model ONCE
    2. Generating 'num_trials' synthetic datasets
    3. Calculating metrics for each generated dataset
    """
    # Get column configuration
    continuous = DATASET_COLUMNS[dataset_name]["continuous"]
    categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    
    # Prepare real data
    X_real = real_df.drop(columns=[target])
    y_real = real_df[target]
    classification = SyntheticDataMetrics.is_classification(y_real)
    
    # Split real data (for utility evaluation)
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )
    
    # Initialize lists to store results
    synth_utility_metrics = []
    synth_attack_metrics = []
    
    print(f"\n--- Training {model_name} (one-time training) ---")
    # Train the synthesizer ONCE
    if model_name in ["dpctgan", "pategan"]:
        synthesizer = Synthesizer.create(model_class, **model_params)
        synthesizer.fit(real_df, preprocessor_eps=0.2)
    elif model_name == "ctgan":
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_df)
        synthesizer = model_class(metadata, **model_params)
        synthesizer.fit(real_df)
    elif model_name == "fctgan" and FCTGAN is not None:
        synthesizer = initialize_fctgan(dataset_name, real_df)
        synthesizer.fit()
    elif model_name == "ctabgan" and CTABGAN is not None:
        synthesizer = initialize_ctabgan(dataset_name, real_df)
        synthesizer.fit()
    else:
        raise ValueError(f"Unknown or unavailable model: {model_name}")
    
    # Generate and evaluate multiple synthetic datasets
    for trial in range(num_trials):
        print(f"\n--- Trial {trial+1}/{num_trials}: Generating and evaluating samples ---")
        
        # Generate synthetic data
        if model_name in ["fctgan", "ctabgan"]:
            synth_data = synthesizer.generate_samples()
        else:
            synth_data = synthesizer.sample(synth_size)
        
        # Skip if target column is missing
        if target not in synth_data.columns:
            print(f"Warning: Target column {target} missing in trial {trial+1}")
            continue
        
        # Prepare synthetic features/target
        X_synth = synth_data.drop(columns=[target])
        y_synth = synth_data[target]
        
        # Calculate statistical metrics
        stats = {
            'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(
                real_df, synth_data, continuous_cols=continuous),
            'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(
                real_df, synth_data, categorical_cols=categorical),
            'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(
                real_df, synth_data, categorical_cols=categorical, continuous_cols=continuous),
            'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(
                real_df[continuous], synth_data[continuous], continuous),
            'JSD': SyntheticDataMetrics.calculate_jsd(
                real_df[categorical], synth_data[categorical], categorical)
        }
        
        # Calculate utility metrics
        utility_metrics = train_model(X_synth, y_synth, X_real_test, y_real_test, classification)
        utility_metrics.update(stats)
        
        # Calculate privacy attack metrics
        attack_metrics = SyntheticDataMetrics.model_inversion_attack(real_df, synth_data, target)
        
        # Store results if both metrics were calculated
        if utility_metrics and attack_metrics:
            synth_utility_metrics.append(utility_metrics)
            synth_attack_metrics.append(attack_metrics)
            metric_runs[(dataset_name, model_name)].append(utility_metrics)
            metric_runs[(dataset_name, model_name)].append(attack_metrics)
            
            # Print trial results
            print(f"\nTrial {trial+1} Results:")
            print("Statistical Metrics:")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}")
            
            print("\nUtility Metrics:")
            for k, v in utility_metrics.items():
                if v is not None:
                    print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            
            print("\nPrivacy Attack Results:")
            for k, v in attack_metrics.items():
                if v is not None:
                    print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
        else:
            print(f"Skipping trial {trial+1} due to missing metrics")
    
    # Return all collected metrics
    return synth_utility_metrics, synth_attack_metrics, metric_runs

def preprocess_data(df, categorical_cols, target_col):
    """Preprocess data by handling missing values and encoding categorical columns"""
    df_clean = df.dropna()
    print(f"Original rows: {len(df)}")
    print(f"Clean rows: {len(df_clean)}")
    print(f"Dropped: {len(df) - len(df_clean)} rows")
    
    # Label encode categorical columns
    for col in categorical_cols:
        if col in df_clean.columns and col != target_col:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    
    # Label encode target if categorical
    if target_col in categorical_cols:
        df_clean[target_col] = LabelEncoder().fit_transform(df_clean[target_col].astype(str))
    
    print("Target distribution:\n", df_clean[target_col].value_counts())
    return df_clean

def evaluate_real_data_baseline( df, target_col,dataset_name):
    """Compute real data utility and attack metrics for baseline"""
    continuous = DATASET_COLUMNS[dataset_name]["continuous"]
    categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    df_clean = preprocess_data(df, categorical, target_col)
    X_real = df_clean.drop(columns=[target_col])
    y_real = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42)

    print(f"\nGetting Real data utility metrics (baseline)")
    real_utility = train_model(
        X_train, y_train, X_test, y_test,
        SyntheticDataMetrics.is_classification(y_real))

    stats = {
        'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(
            df_clean, df_clean, continuous_cols=continuous),
        'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(
            df_clean, df_clean, categorical_cols=categorical),
        'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(
            df_clean, df_clean, categorical_cols=categorical, continuous_cols=continuous),
        'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(
            df_clean[continuous], df_clean[continuous], continuous),
        'JSD': SyntheticDataMetrics.calculate_jsd(
            df_clean[categorical], df_clean[categorical], categorical)
    }
    real_utility.update(stats)
    print(f"\nReal data utility metrics (baseline)", real_utility)

    print(f"\nGetting Real data attack metrics (baseline)")
    real_attack = SyntheticDataMetrics.model_inversion_attack(df_clean, df_clean, target_col)
    print(f"\nReal data attack metrics (baseline)", real_attack)

    return df_clean,real_utility, real_attack

def print_pvalue_guidelines():
    message = """
    === P-VALUE GUIDELINES FOR SYNTHETIC DATA EVALUATION ===

    1. Utility & Statistical Metrics (e.g., Accuracy, F1-score, AUROC, PearsonCorrDiff, Wasserstein, JSD):
    - Goal: Synthetic data metrics should be CLOSE to real data metrics.
    - Interpretation: 
        * p-value >= 0.05  --> No statistically significant difference (GOOD)
        * p-value <  0.05  --> Statistically significant difference (Synthetic data differs from real)

    2. Privacy Attack Metrics (e.g., Attack Accuracy, Attack AUROC, MSE for attacks):
    - Goal: Synthetic data should be SAFER than real data (harder to attack).
    - Interpretation:
        * p-value < 0.05  --> Significant difference, synthetic data offers better privacy (GOOD)
        * p-value >= 0.05 --> No significant difference in privacy (NOT ideal)

    Summary:
    --------------------------------------------------------
    Metric Type             | Ideal p-value           | Meaning
    --------------------------------------------------------
    Utility & Statistical   | p >= 0.05               | Synthetic ≈ Real (desired)
    Privacy Attack          | p < 0.05                | Synthetic safer than Real (desired)
    --------------------------------------------------------

    Note: p-value is the probability that observed differences are due to chance under the null hypothesis.
    Lower p-value means stronger evidence against the null hypothesis (i.e., real and synthetic differ).

    """
    print(message)
def save_metrics_with_error_bars(metrics_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, metric_list in metrics_dict.items():
        df = pd.DataFrame(metric_list)

        # Save full metrics
        full_path = os.path.join(output_dir, f"{name}_full.csv")
        df.to_csv(full_path, index=False)

        # Calculate mean and std
        mean = df.mean(numeric_only=True)
        std = df.std(numeric_only=True)

        # Combine mean ± std into a single row for readability
        summary = {
            metric: f"{mean[metric]:.4f} ± {std[metric]:.4f}" for metric in mean.index
        }

        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, f"{name}_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"Saved: {full_path} and {summary_path}")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict



def save_results(all_results, metric_runs):
    os.makedirs("Experiment_Results", exist_ok=True)
    
    fd_metrics = []
    stat_sim = []
    privacy_metrics = []
    ml_utility = []
    std_dict = defaultdict(dict)

    for result in all_results:
        dataset = result['dataset']
        model = result['model']
        utility = result['utility_metrics']
        attack = result['attack_metrics']
        utility_pvals = result['utility_p_values']
        attack_pvals = result['attack_p_values']

        # Get std from metric_runs DataFrame (multiple runs per dataset+model)
        runs = pd.DataFrame(metric_runs[(dataset, model)])
        numeric_cols = runs.select_dtypes(include='number').columns
        stds = runs[numeric_cols].std().to_dict()
        std_dict[(dataset, model)] = stds

        fd_metrics.append({
            'dataset': dataset,
            'model': model,
            'PearsonCorrDiff': utility.get('PearsonCorrDiff'),
            'PearsonCorrDiff_std': stds.get('PearsonCorrDiff'),
            'PearsonCorrDiff_p': utility_pvals.get('PearsonCorrDiff'),
            'UncertaintyCoeffDiff': utility.get('UncertaintyCoeffDiff'),
            'UncertaintyCoeffDiff_std': stds.get('UncertaintyCoeffDiff'),
            'UncertaintyCoeffDiff_p': utility_pvals.get('UncertaintyCoeffDiff'),
            'CorrelationRatioDiff': utility.get('CorrelationRatioDiff'),
            'CorrelationRatioDiff_std': stds.get('CorrelationRatioDiff'),
            'CorrelationRatioDiff_p': utility_pvals.get('CorrelationRatioDiff')
        })

        stat_sim.append({
            'dataset': dataset,
            'model': model,
            'JSD': utility.get('JSD'),
            'JSD_p': utility_pvals.get('JSD'),
            'JSD_std': stds.get('JSD'),
            'Wasserstein': utility.get('Wasserstein'),
            'Wasserstein_std': stds.get('Wasserstein'),
            'Wasserstein_p': utility_pvals.get('Wasserstein')
        })

        privacy_metrics.append({
            'dataset': dataset,
            'model': model,
            'MIA_Accuracy': attack.get('Accuracy'),
            'MIA_Accuracy_std': stds.get('Accuracy'),
            'MIA_Accuracy_p': attack_pvals.get('Accuracy'),
            'MIA_AUROC': attack.get('AUROC'),
            'MIA_AUROC_std': stds.get('AUROC'),
            'MIA_AUROC_p': attack_pvals.get('AUROC')
        })

        ml_utility.append({
            'dataset': dataset,
            'model': model,
            'F1-score': utility.get('F1-score'),
            'F1-score_p': utility_pvals.get('F1-score'),
            'F1-score_std': stds.get('F1-score'),
            'AUROC': utility.get('AUROC'),
            'AUROC_std': stds.get('AUROC'),
            'AUROC_p': utility_pvals.get('AUROC'),
            'AUPRC': utility.get('AUPRC'),
            'AUPRC_std': stds.get('AUPRC'),
            'AUPRC_p': utility_pvals.get('AUPRC'),
            'Accuracy': utility.get('Accuracy'),
            'Accuracy_std': stds.get('Accuracy'),
            'Accuracy_p': utility_pvals.get('Accuracy')
        })

    metrics = {
        'fd_metrics': fd_metrics,
        'stat_sim': stat_sim,
        'privacy_metrics': privacy_metrics,
        'ml_utility': ml_utility
    }

    # Save CSV files with rounded numeric values
    for name, data in metrics.items():
        df = pd.DataFrame(data)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(4)
        csv_path = f"Experiment_Results/{name}_with_pvalues.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    # After saving CSVs, generate plots for each metric group
    generate_plots_for_metrics(metrics)

import seaborn as sns
import matplotlib.pyplot as plt


def generate_plots_for_metrics(metrics_dict):
    os.makedirs("Experiment_Results/plots", exist_ok=True)
    # Fixed color mapping for datasets
    # Define colors for datasets
    dataset_colors = {
        'adult': '#1f77b4',   # Blue
        'credit': '#ff7f0e'   # Orange
    }
    for group_name, metric_list in metrics_dict.items():
        df = pd.DataFrame(metric_list)
        
        # Determine metric columns (exclude dataset, model, and p-values)
        metric_cols = [col for col in df.columns if col not in ['dataset', 'model'] and not col.endswith('_p') and not col.endswith('_std')]
        
        print(f"Plotting metrics from group: {group_name} ({len(metric_cols)} metrics)")

        for metric in metric_cols:
            std_col = f"{metric}_std"
            if std_col not in df.columns:
                df[std_col] = 0.0  # If std not available, use zero

            # Create seaborn-friendly format
            plot_df = df[['dataset', 'model', metric, std_col]].copy()
            plot_df = plot_df.rename(columns={
                metric: "value",
                std_col: "std"
            })

            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")

            ax = sns.barplot(
                data=plot_df,
                x="model",
                y="value",
                hue="dataset",
                palette=dataset_colors,
                errorbar=None,
                capsize=0.1,
                errwidth=1.5
            )

            # Add error bars manually
            for i in range(len(plot_df)):
                row = plot_df.iloc[i]
                xpos = i % len(plot_df['model'].unique())
                group_offset = list(plot_df['dataset'].unique()).index(row['dataset'])
                ax.errorbar(
                    xpos + group_offset * 0.1 - 0.1,
                    row['value'],
                    yerr=row['std'],
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

            ax.set_title(f"{metric} ({group_name})", fontsize=14)
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend_.remove()  # Hide legend here (saved separately)
            plt.tight_layout()

            filename = f"Experiment_Results/plots/{group_name}_{metric}.pdf".replace(" ", "_")
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")

    save_dataset_legend(dataset_colors)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def save_dataset_legend(dataset_colors):
    sns.set_theme(style="whitegrid")  # seaborn styling

    plt.figure(figsize=(4, 2))

    # Create legend handles manually using Line2D objects with markers only
    handles = [
        Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=dataset)
        for dataset, color in dataset_colors.items()
    ]

    plt.legend(handles=handles, title="Dataset", ncol=2, loc='center', frameon=False)
    plt.axis('off')

    filename = "Experiment_Results/plots/dataset_legend.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset legend: {filename}")


#print_pvalue_guidelines()
# Differential privacy parameters
pategan_params = dict(
    epsilon=1.0,
    delta=1e-5,
    binary=True,
    latent_dim=64,
    batch_size=64,
    teacher_iters=10,
    student_iters=5
)

dpctgan_params = dict(
    epsilon=2.0,
    delta=1e-6,
    batch_size=128,
    epochs=150
)
# Initialize dictionaries to store all results
all_fd_metrics = []
all_stat_sim = []
all_privacy_metrics = []
all_ml_utility = []
num_trials = 2
all_results = []
# Fixed color mapping for datasets
dataset_colors = {
    "Adult": "tab:blue",
    "Creditrisk": "tab:orange"
}
#metric_runs = defaultdict(lambda: defaultdict(list))  # key = (dataset, model), value = list of metric dicts
metric_runs = defaultdict(list)
# Evaluate all available models
models_to_evaluate = [
    #("ctgan", CTGANSynthesizer, {'epochs': 150}),
    #("pategan", PATEGAN, pategan_params),
    ("dpctgan", DPCTGAN, dpctgan_params),
    #("fctgan", FCTGAN, {}),
    #("ctabgan", CTABGAN, {}),
]
# Main execution
for dataset_name in ["adult","credit"]:
    print(f"\n=== Dataset: {dataset_name.upper()} ===")
    df, target_col = load_dataset(dataset_name)
    
    df_clean, real_utility, real_attack = evaluate_real_data_baseline(df,target_col,dataset_name)
    # Convert single real evaluation to list for p-value calculation
    real_utility_list = [real_utility]
    real_attack_list = [real_attack]
    # Preprocess data
    # df_clean = df.dropna()
    # print(f"Original rows: {len(df)}")
    # print(f"Clean rows: {len(df_clean)}")
    # print(f"Dropped: {len(df) - len(df_clean)} rows")
    
    # Get columns for this dataset
    categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    
    # Label encode categorical columns
    label_encoder = LabelEncoder()
    for col in categorical:
        if col in df_clean.columns:
            df_clean[col] = label_encoder.fit_transform(df_clean[col])
    
    print("Target distribution:\n", df_clean[target_col].value_counts())
        
    # if FCTGAN is not None:
    #     models_to_evaluate.append(("fctgan", FCTGAN, {}))
    
    # if CTABGAN is not None:
    #     models_to_evaluate.append(("ctabgan", CTABGAN, {}))
    for model_name, model_class, model_params in models_to_evaluate:
        print(f"\nEvaluating model: {model_name}")

        synth_utility_metrics = []
        synth_attack_metrics = []

        # for trial in range(num_trials):
        # print(f"Trial {trial + 1}/{num_trials}")
        synth_utility_metrics, synth_attack_metrics,metric_runs = evaluate_synthetic_data_model(
            df_clean, target_col, synth_size=1000, dataset_name=dataset_name,
            model_name=model_name, model_class=model_class, model_params=model_params,num_trials=10
        )

            # if utility_metrics and attack_metrics:
            #     synth_utility_metrics.append(utility_metrics)
            #     synth_attack_metrics.append(attack_metrics)
            #     metric_runs[(dataset_name, model_name)].append(utility_metrics)
            #     metric_runs[(dataset_name, model_name)].append(attack_metrics)

        if synth_utility_metrics:
            utility_df = pd.DataFrame(synth_utility_metrics).select_dtypes(include='number')
            attack_df = pd.DataFrame(synth_attack_metrics).select_dtypes(include='number')

            mean_utility = utility_df.mean().to_dict()
            mean_attack = attack_df.mean().to_dict()

            utility_p_values = SyntheticDataMetrics.calculate_p_values(
                real_utility_list * len(synth_utility_metrics), synth_utility_metrics
            )
            attack_p_values = SyntheticDataMetrics.calculate_p_values(
                real_attack_list * len(synth_attack_metrics), synth_attack_metrics
            )

            all_results.append({
                'dataset': dataset_name,
                'model': model_name,
                'utility_metrics': mean_utility,
                'attack_metrics': mean_attack,
                'utility_p_values': utility_p_values,
                'attack_p_values': attack_p_values,
                'real_utility': real_utility,
                'real_attack': real_attack
            })

save_results(all_results, metric_runs)
    # for model_name, model_class, model_params in models_to_evaluate:
    #     utility_metrics, attack_metrics = evaluate_synthetic_data_model(
    #         df_clean, target_col, synth_size=1000, dataset_name=dataset_name,
    #         model_name=model_name, model_class=model_class, model_params=model_params
    #     )
    #     if attack_metrics:
    #         attack_results_dict[model_name] = attack_metrics
    #     if utility_metrics and attack_metrics:
    #             # Prepare data for each CSV file
                
    #             # 1. FD Metrics
    #             fd_metrics = {
    #                 'dataset': dataset_name,
    #                 'model': model_name,
    #                 'PearsonCorrDiff': utility_metrics.get('PearsonCorrDiff'),
    #                 'UncertaintyCoeffDiff': utility_metrics.get('UncertaintyCoeffDiff'),
    #                 'CorrelationRatioDiff': utility_metrics.get('CorrelationRatioDiff')
    #             }
    #             all_fd_metrics.append(fd_metrics)
                
    #             # 2. Stat Sim
    #             stat_sim = {
    #                 'dataset': dataset_name,
    #                 'model': model_name,
    #                 'JSD': utility_metrics.get('JSD'),
    #                 'Wasserstein': utility_metrics.get('Wasserstein')
    #             }
    #             all_stat_sim.append(stat_sim)
                
    #             # 3. Privacy Metrics
    #             privacy_metrics = {
    #                 'dataset': dataset_name,
    #                 'model': model_name,
    #                 'MIA_Accuracy': attack_metrics.get('Accuracy'),
    #                 'MIA_AUROC': attack_metrics.get('AUROC'),
    #                 'MIA_MSE': attack_metrics.get('MSE'),
    #                 'MIA_R2': attack_metrics.get('R2'),
    #                 'Membership inference attack AUROC': attack_metrics.get('Membership inference attack AUROC')
    #             }
    #             all_privacy_metrics.append(privacy_metrics)
                
    #             # 4. Machine Learning Utility
    #             ml_utility = {
    #                 'dataset': dataset_name,
    #                 'model': model_name,
    #                 'F1-score': utility_metrics.get('F1-score'),
    #                 'AUROC': utility_metrics.get('AUROC'),
    #                 'AUPRC': utility_metrics.get('AUPRC'),
    #                 'Accuracy': utility_metrics.get('Accuracy')
    #             }
    #             all_ml_utility.append(ml_utility)
    
    
    # # Convert to DataFrames and save as CSV
    # fd_metrics_df = pd.DataFrame(all_fd_metrics)
    # stat_sim_df = pd.DataFrame(all_stat_sim)
    # privacy_metrics_df = pd.DataFrame(all_privacy_metrics)
    # ml_utility_df = pd.DataFrame(all_ml_utility)
    
    # # Function to round numeric columns to 2 decimal places
    # def round_numeric_columns(df):
    #     for col in df.columns:
    #         if pd.api.types.is_numeric_dtype(df[col]):
    #             df[col] = df[col].round(2)
    #     return df
    #  # Round all numeric values in each DataFrame
    # fd_metrics_df = round_numeric_columns(fd_metrics_df)
    # stat_sim_df = round_numeric_columns(stat_sim_df)
    # privacy_metrics_df = round_numeric_columns(privacy_metrics_df)
    # ml_utility_df = round_numeric_columns(ml_utility_df)
    
    # # Save to CSV files
    # fd_metrics_df.to_csv('fd_metrics.csv', index=False)
    # stat_sim_df.to_csv('stat_sim.csv', index=False)
    # privacy_metrics_df.to_csv('privacy_metrics.csv', index=False)
    # ml_utility_df.to_csv('machine_learning_utility.csv', index=False)
    
    # print("\nResults saved to CSV files:")
    # print("- fd_metrics.csv")
    # print("- stat_sim.csv")
    # print("- privacy_metrics.csv")
    # print("- machine_learning_utility.csv")