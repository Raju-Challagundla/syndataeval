# -*- coding: utf-8 -*-
"""Experiment_Script_Adult.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GBQJQEbBKPYrlEffMrIQa9EGUpnK9Vr7
"""

#import sys
#sys.path.append('/content/FCT-GAN')

from model.fctgan import FCTGAN
from model.eval.evaluation_original import get_utility_metrics,stat_sim,privacy_metrics
import numpy as np
import pandas as pd
import glob
import torch
from scipy.stats import ks_2samp


print(torch.version.cuda)
# Check if CUDA is available and use it if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_exp = 1
dataset = "Adult"
real_path = "Real_Datasets/Adult.csv"
fake_file_root = "Fake_Datasets"
results_path = "results"

synthesizer =  FCTGAN(
                 dataset=dataset,
                 raw_csv_path = real_path,
                 test_ratio = 0.20,
                 categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 epochs=150)

for i in range(num_exp):
    synthesizer.fit()
    syn = synthesizer.generate_samples()
    syn.to_csv(f"{fake_file_root}/{dataset}/{dataset}_fake_{i}.csv", index= False)
#fake_paths = glob.glob(fake_file_root+"/"+dataset+"/"+"*")

#model_dict = {"Classification":["lr","dt","rf","mlp","svm"]}
#result_mat = get_utility_metrics(real_path,fake_paths,"MinMax",model_dict, test_ratio = 0.20)
print("Getting metrics")

# Load the real dataset for comparison
data = pd.read_csv(real_path)

# Specify continuous (numerical) columns
continuous_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

# Calculate metrics for each numerical column
for column in continuous_columns:
    # Real data metrics
    real_mean = data[column].mean()
    real_std = data[column].std()

    # Synthetic data metrics
    synthetic_mean = syn[column].mean()
    synthetic_std = syn[column].std()

    # Kolmogorov-Smirnov Statistic (for distribution comparison)
    ks_stat, _ = ks_2samp(data[column], syn[column])

    # Output the results
    print(f"Column: {column}")
    print(f"Real Data Mean: {real_mean}")
    print(f"Synthetic Data Mean: {synthetic_mean}")
    print(f"Real Data Std Dev: {real_std}")
    print(f"Synthetic Data Std Dev: {synthetic_std}")
    print(f"Kolmogorov-Smirnov Statistic: {ks_stat}")
    print("-" * 40)
syn
print("Metrics calculated from synthetic data generated by the trained synthesizer.")


# Save the synthetic data to a CSV file
syn.to_csv('synthetic_adult_data_fctgan.csv', index=False)

print("Synthetic data saved to 'synthetic_adult_data_fctgan.csv'")
