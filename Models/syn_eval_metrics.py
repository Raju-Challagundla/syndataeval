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

def evaluate_synthetic_data_model(real_df, target, synth_size, dataset_name, model_name, model_class, model_params):
    """
    Evaluate synthetic data generated by CTGAN, FCT-GAN or CTABGAN
    Returns utility metrics and attack metrics
    """
    # Get columns for this dataset
    continuous = DATASET_COLUMNS[dataset_name]["continuous"]
    categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    
    # Prepare real data
    X_real = real_df.drop(columns=[target])
    y_real = real_df[target]
    classification = SyntheticDataMetrics.is_classification(y_real)
    
    # Split real data for evaluation
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )
    print(f"\n--- Training {model_name} on {dataset_name} dataset ---")
    # Initialize and train synthesizer
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
    
    
    # Train synthesizer
    # if model_name == "ctgan":
    #     synthesizer.fit(real_df)
    # elif model_name == "fctgan":
    #     synthesizer.fit(real_df, None)
    # elif model_name == "ctabgan":
    #     synthesizer.fit(real_df)
    print(f"\n--- Generating Samples ---")
    # Generate synthetic data
    if model_name in ["fctgan", "ctabgan"]:
        synth_data = synthesizer.generate_samples()
    else:
        synth_data = synthesizer.sample(synth_size)

    
    # Prepare synthetic features/target
    if target not in synth_data.columns:
        print(f"Warning: Target column {target} not found in synthetic data")
        return None, None
    
    X_synth = synth_data.drop(columns=[target])
    y_synth = synth_data[target]
    
    # Calculate statistical metrics
    stats = {
        'PearsonCorrDiff': SyntheticDataMetrics.pearson_correlation_difference(real_df, synth_data, continuous_cols=continuous),
        'UncertaintyCoeffDiff': SyntheticDataMetrics.uncertainty_coefficient_difference(real_df, synth_data, categorical_cols=categorical),
        'CorrelationRatioDiff': SyntheticDataMetrics.correlation_ratio_difference(real_df, synth_data, categorical_cols=categorical, continuous_cols=continuous),
        'Wasserstein': SyntheticDataMetrics.calculate_wasserstein(real_df[continuous], synth_data[continuous], continuous),
        'JSD': SyntheticDataMetrics.calculate_jsd(real_df[categorical], synth_data[categorical], categorical)
    }
    
    # Utility evaluation
    utility_metrics = train_model(X_synth, y_synth, X_real_test, y_real_test, classification)
    utility_metrics.update(stats)
    
    print("\nStatistical Metrics:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    
    print("\nUtility Metrics:")
    for k, v in utility_metrics.items():
        if v is not None:
            try:
                # Try float formatting if possible
                print(f"{k}: {float(v):.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            except (ValueError, TypeError):
                # Fallback to regular string output
                print(f"{k}: {v}")
    
    # Privacy attacks
    mi_metrics = SyntheticDataMetrics.model_inversion_attack(real_df, synth_data, target)
    meminf_metrics = SyntheticDataMetrics.membership_inference_attack(real_df, synth_data, target)
    attack_metrics = {**mi_metrics, **meminf_metrics}
    
    print("\nPrivacy Attack Results:")
    for k, v in attack_metrics.items():
        if v is None:
            print(f"  {k}: Not Applicable")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save model
    model_path = os.path.join(pkl_dir, f"{model_name}_{dataset_name}.pkl")
    joblib.dump(synthesizer, model_path)
    print(f"\nModel saved to {model_path}")
    
    return utility_metrics, attack_metrics

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
    epsilon=1.0,
    delta=1e-5,
    batch_size=64,
    epochs=150
)
# Initialize dictionaries to store all results
all_fd_metrics = []
all_stat_sim = []
all_privacy_metrics = []
all_ml_utility = []

# Main execution
for dataset_name in ["adult","credit"]:
    print(f"\n=== Dataset: {dataset_name.upper()} ===")
    df, target_col = load_dataset(dataset_name)
    
    # Preprocess data
    df_clean = df.dropna()
    print(f"Original rows: {len(df)}")
    print(f"Clean rows: {len(df_clean)}")
    print(f"Dropped: {len(df) - len(df_clean)} rows")
    
    # Get columns for this dataset
    categorical = DATASET_COLUMNS[dataset_name]["categorical"]
    
    # Label encode categorical columns
    label_encoder = LabelEncoder()
    for col in categorical:
        if col in df_clean.columns:
            df_clean[col] = label_encoder.fit_transform(df_clean[col])
    
    print("Target distribution:\n", df_clean[target_col].value_counts())
    
    attack_results_dict = {}
    
    # Evaluate all available models
    models_to_evaluate = [
        #("ctgan", CTGANSynthesizer, {'epochs': 150}),
        ("pategan", PATEGAN, pategan_params),
        #("dpctgan", DPCTGAN, dpctgan_params),
        #("fctgan", FCTGAN, {}),
        #("ctabgan", CTABGAN, {}),
    ]
    
    # if FCTGAN is not None:
    #     models_to_evaluate.append(("fctgan", FCTGAN, {}))
    
    # if CTABGAN is not None:
    #     models_to_evaluate.append(("ctabgan", CTABGAN, {}))

    for model_name, model_class, model_params in models_to_evaluate:
        utility_metrics, attack_metrics = evaluate_synthetic_data_model(
            df_clean, target_col, synth_size=1000, dataset_name=dataset_name,
            model_name=model_name, model_class=model_class, model_params=model_params
        )
        if attack_metrics:
            attack_results_dict[model_name] = attack_metrics
        if utility_metrics and attack_metrics:
                # Prepare data for each CSV file
                
                # 1. FD Metrics
                fd_metrics = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'PearsonCorrDiff': utility_metrics.get('PearsonCorrDiff'),
                    'UncertaintyCoeffDiff': utility_metrics.get('UncertaintyCoeffDiff'),
                    'CorrelationRatioDiff': utility_metrics.get('CorrelationRatioDiff')
                }
                all_fd_metrics.append(fd_metrics)
                
                # 2. Stat Sim
                stat_sim = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'JSD': utility_metrics.get('JSD'),
                    'Wasserstein': utility_metrics.get('Wasserstein')
                }
                all_stat_sim.append(stat_sim)
                
                # 3. Privacy Metrics
                privacy_metrics = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'MIA_Accuracy': attack_metrics.get('Accuracy'),
                    'MIA_AUROC': attack_metrics.get('AUROC'),
                    'MIA_MSE': attack_metrics.get('MSE'),
                    'MIA_R2': attack_metrics.get('R2'),
                    'Membership inference attack AUROC': attack_metrics.get('Membership inference attack AUROC')
                }
                all_privacy_metrics.append(privacy_metrics)
                
                # 4. Machine Learning Utility
                ml_utility = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'F1-score': utility_metrics.get('F1-score'),
                    'AUROC': utility_metrics.get('AUROC'),
                    'AUPRC': utility_metrics.get('AUPRC'),
                    'Accuracy': utility_metrics.get('Accuracy')
                }
                all_ml_utility.append(ml_utility)
    
    
    # Convert to DataFrames and save as CSV
    fd_metrics_df = pd.DataFrame(all_fd_metrics)
    stat_sim_df = pd.DataFrame(all_stat_sim)
    privacy_metrics_df = pd.DataFrame(all_privacy_metrics)
    ml_utility_df = pd.DataFrame(all_ml_utility)
    
    # Function to round numeric columns to 2 decimal places
    def round_numeric_columns(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
        return df
     # Round all numeric values in each DataFrame
    fd_metrics_df = round_numeric_columns(fd_metrics_df)
    stat_sim_df = round_numeric_columns(stat_sim_df)
    privacy_metrics_df = round_numeric_columns(privacy_metrics_df)
    ml_utility_df = round_numeric_columns(ml_utility_df)
    
    # Save to CSV files
    fd_metrics_df.to_csv('fd_metrics.csv', index=False)
    stat_sim_df.to_csv('stat_sim.csv', index=False)
    privacy_metrics_df.to_csv('privacy_metrics.csv', index=False)
    ml_utility_df.to_csv('machine_learning_utility.csv', index=False)
    
    print("\nResults saved to CSV files:")
    print("- fd_metrics.csv")
    print("- stat_sim.csv")
    print("- privacy_metrics.csv")
    print("- machine_learning_utility.csv")