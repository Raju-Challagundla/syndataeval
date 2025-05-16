import pandas as pd
from tqdm import tqdm
from model.fctgan import FCTGAN  # Assuming FCTGAN is available from this import
import joblib
import numpy as np
#from dython.nominal import uncertainty_coefficient
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import pearsonr

# Define the columns
categorical = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'loan_status',
    'cb_person_default_on_file'
]

continuous = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

mixed_pairs = [
    ('loan_grade', 'loan_int_rate'),
    ('person_home_ownership', 'person_income'),
    ('loan_intent', 'loan_amnt')
]
mixed_columns = {
    'loan_grade': ['loan_int_rate'],
    'person_home_ownership': ['person_income'],
    'loan_intent': ['loan_amnt']
}
# Load the data
# Replace this with the actual path to your CSV data file
real_path = "../../Real_Datasets/credit_risk_dataset.csv"
real_data = pd.read_csv(real_path)

# Preprocessing: Encoding categorical columns
encoder = LabelEncoder()
for col in categorical:
    real_data[col] = encoder.fit_transform(real_data[col])

# Split the data into continuous and categorical
data_clean = real_data.copy()

# Initialize the FCTGAN model
synthesizer = FCTGAN(dataset="CreditRisk",
                     raw_csv_path=real_path,
                     test_ratio=0.20,
                     categorical_columns=categorical,
                     log_columns=[],
                     mixed_columns=mixed_columns,
                     integer_columns=continuous,
                     problem_type={"Classification": 'loan_status'},epochs=150)  # Adjust to your problem type

# Train the model for multiple epochs
num_exp = 1  # Set number of epochs
for epoch in tqdm(range(num_exp), desc="Training Epochs"):
    synthesizer.fit()

# Save the trained model
joblib.dump(synthesizer, 'fctgan_model.pkl')

# Generate synthetic data after training
synthetic_data = synthesizer.generate_samples()

# Metric Calculations
from dython.nominal import theils_u

def uncertainty_coefficient(x, y):
    return theils_u(x, y)
def pearson_corr_diff(real, synth, continuous_cols):
    real_corr = real[continuous_cols].corr(method='pearson')
    synth_corr = synth[continuous_cols].corr(method='pearson')
    diff = (real_corr - synth_corr).abs().mean().mean()
    return diff
def avg_uncertainty_diff(real, synth, cat_cols):
    total_diff = 0
    count = 0
    for i in cat_cols:
        for j in cat_cols:
            if i != j:
                u_real = uncertainty_coefficient(real[i], real[j])
                u_synth = uncertainty_coefficient(synth[i], synth[j])
                total_diff += abs(u_real - u_synth)
                count += 1
    return total_diff / count
import numpy as np

def correlation_ratio(categories, measurements):
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
    total_diff = 0
    count = 0
    for cat in cat_cols:
        for num in num_cols:
            r = correlation_ratio(real[cat], real[num])
            s = correlation_ratio(synth[cat], synth[num])
            total_diff += abs(r - s)
            count += 1
    return total_diff / count

pearson_score = pearson_corr_diff(real_data, synthetic_data, continuous)
uncertainty_score = avg_uncertainty_diff(real_data, synthetic_data, categorical)
corr_ratio_score = avg_corr_ratio_diff(real_data, synthetic_data, categorical, continuous)

print(f"Pearson Corr Diff: {pearson_score:.4f}")
print(f"Uncertainty Coefficient Diff: {uncertainty_score:.4f}")
print(f"Correlation Ratio Diff: {corr_ratio_score:.4f}")

