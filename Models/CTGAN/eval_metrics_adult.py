import sys
import sdv
print(sdv.version.public)
import torch

if torch.cuda.is_available():
    print("CUDA available. Running on GPU.")
else:
    print("CUDA not available. Running on CPU.")

import pandas as pd

real_data = pd.read_csv("../../Real_Datasets/Adult_dataset.csv")

real_data.head()

categorical = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native-country',
    'income'
]
continuous = [
    'age',
    'fnlwgt',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

data_clean = real_data.dropna()
print(f"Original rows: {len(real_data)}")
print(f"Clean rows:    {len(data_clean)}")
print(f"Dropped:       {len(real_data) - len(data_clean)} rows")

from sklearn.preprocessing import LabelEncoder

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns
for col in categorical:
    real_data[col] = label_encoder.fit_transform(real_data[col])

from tqdm import tqdm
import joblib

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# write a metadata object to describe the data, primary key, etc.
# this can be done manually or you can auto-detect/update it
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_clean)
# Initialize CTGAN with 150 epochs
ctgan = CTGANSynthesizer(metadata, epochs=150)

# Training with epoch logging and loss tracking
for epoch in tqdm(range(1), desc="Training Epochs"):
    ctgan.fit(data_clean)
    print(f"Epoch {epoch + 1}/{150}")

# Save it
joblib.dump(ctgan, 'ctgan_1.20.0_adult.pkl')
print(".pkl saved")

# After training is complete, generate synthetic data
synthetic_data = ctgan.sample(len(real_data))

def pearson_corr_diff(real, synth, continuous_cols):
    real_corr = real[continuous_cols].corr(method='pearson')
    synth_corr = synth[continuous_cols].corr(method='pearson')
    diff = (real_corr - synth_corr).abs().mean().mean()
    return diff

from dython.nominal import theils_u

def uncertainty_coefficient(x, y):
    return theils_u(x, y)

import dython
from dython.nominal import conditional_entropy

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
print("Printing Stats...")
print(f"Pearson Corr Diff: {pearson_score:.4f}")
print(f"Uncertainty Coefficient Diff: {uncertainty_score:.4f}")
print(f"Correlation Ratio Diff: {corr_ratio_score:.4f}")
