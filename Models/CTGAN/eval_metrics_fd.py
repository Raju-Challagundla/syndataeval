import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata
from dython.nominal import theils_u

# Hardcoded dataset configurations
DATASET_CONFIGS = {
    'cr': {  # credit risk
        'path': 'Real_Datasets/credit_risk_dataset.csv',
        'categorical': [
            'person_home_ownership',
            'loan_intent',
            'loan_grade',
            'loan_status',
            'cb_person_default_on_file'
        ],
        'continuous': [
            'person_age',
            'person_income',
            'person_emp_length',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length'
        ]
    },
    'adult': {
        'path': 'Real_Datasets/adult_dataset.csv',
        'categorical': [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native-country',
            'income'
        ],
        'continuous': [
            'age',
            'fnlwgt',
            'education-num',
            'capital-gain',
            'capital-loss',
            'hours-per-week'
        ]
    }
}

def load_dataset(dataset_name):
    """Load and preprocess the dataset based on predefined configurations."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    data = pd.read_csv(config['path'])
    
    # Encode categorical columns (must match how training data was processed)
    encoder = LabelEncoder()
    for col in config['categorical']:
        if col in data.columns:
            data[col] = encoder.fit_transform(data[col].astype(str))
    
    return data, config['categorical'], config['continuous']

def load_model(model_path):
    """Load a trained model from pickle file."""
    return joblib.load(model_path)

def generate_synthetic_data(model, num_rows):
    """Generate synthetic data from the loaded model."""
    return model.sample(num_rows=num_rows)

def uncertainty_coefficient(x, y):
    """Calculate Theil's U (Uncertainty Coefficient)."""
    return theils_u(x, y)

def pearson_corr_diff(real, synth, continuous_cols):
    """Calculate the difference in Pearson correlations."""
    real_corr = real[continuous_cols].corr(method='pearson')
    synth_corr = synth[continuous_cols].corr(method='pearson')
    diff = (real_corr - synth_corr).abs().mean().mean()
    return diff

def avg_uncertainty_diff(real, synth, cat_cols):
    """Calculate average difference in uncertainty coefficients."""
    total_diff = 0
    count = 0
    for i in cat_cols:
        for j in cat_cols:
            if i != j:
                u_real = uncertainty_coefficient(real[i], real[j])
                u_synth = uncertainty_coefficient(synth[i], synth[j])
                total_diff += abs(u_real - u_synth)
                count += 1
    return total_diff / count if count > 0 else 0

def correlation_ratio(categories, measurements):
    """Calculate correlation ratio between categorical and continuous variables."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg = np.mean(measurements)
    numerator = 0
    denominator = 0
    for i in range(cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        if len(cat_measures) > 0:
            numerator += len(cat_measures) * (np.mean(cat_measures) - y_avg) ** 2
    denominator = np.sum((measurements - y_avg) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else 0

def avg_corr_ratio_diff(real, synth, cat_cols, num_cols):
    """Calculate average difference in correlation ratios."""
    total_diff = 0
    count = 0
    for cat in cat_cols:
        for num in num_cols:
            r = correlation_ratio(real[cat], real[num])
            s = correlation_ratio(synth[cat], synth[num])
            total_diff += abs(r - s)
            count += 1
    return total_diff / count if count > 0 else 0

def evaluate_synthetic_data(real_data, synthetic_data, categorical_cols, continuous_cols):
    """Evaluate the quality of synthetic data."""
    pearson_score = pearson_corr_diff(real_data, synthetic_data, continuous_cols)
    uncertainty_score = avg_uncertainty_diff(real_data, synthetic_data, categorical_cols)
    corr_ratio_score = avg_corr_ratio_diff(real_data, synthetic_data, categorical_cols, continuous_cols)
    
    return {
        'pearson_corr_diff': pearson_score,
        'uncertainty_coefficient_diff': uncertainty_score,
        'correlation_ratio_diff': corr_ratio_score
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate pre-trained synthetic data models.')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model .pkl file')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['cr', 'adult'],
                      help='Dataset to use (cr for credit risk, adult for adult census)')
    
    args = parser.parse_args()
    
    # Load the original dataset
    real_data, categorical_cols, continuous_cols = load_dataset(args.dataset)
    
    # Load the pre-trained model
    model = load_model(args.model_path)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(model, len(real_data))
    
    # Evaluate
    evaluation_results = evaluate_synthetic_data(
        real_data, synthetic_data, categorical_cols, continuous_cols
    )
    
    print("\nEvaluation Results:")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Pearson Correlation Difference: {evaluation_results['pearson_corr_diff']:.4f}")
    print(f"Uncertainty Coefficient Difference: {evaluation_results['uncertainty_coefficient_diff']:.4f}")
    print(f"Correlation Ratio Difference: {evaluation_results['correlation_ratio_diff']:.4f}")

if __name__ == "__main__":
    main()
