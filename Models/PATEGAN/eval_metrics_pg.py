import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, mean_absolute_percentage_error, r2_score, explained_variance_score
)
from snsynth import Synthesizer
from snsynth.pytorch.nn.pategan import PATEGAN

import joblib
from datetime import datetime

def load_dataset(name):
    if name == "adult":
        data = pd.read_csv("Real_Datasets/Adult_dataset.csv")
        target = 'income'
    elif name == "credit":
        data = pd.read_csv("Real_Datasets/credit_risk_dataset.csv")
        target = 'loan_status'
    else:
        raise ValueError("Unknown dataset")
    data = pd.get_dummies(data, drop_first=True)
    return data, target

def is_classification(y):
    return y.dtype == 'object' or len(y.unique()) <= 10

def train_model(X_train, y_train, X_test, y_test, classification=True):
    if classification:
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else None

        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred, average='weighted'),
            'AUROC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            'AUPRC': average_precision_score(y_test, y_proba) if y_proba is not None else None
        }
    else:
        reg = RandomForestRegressor(random_state=0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        return {
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'EVS': explained_variance_score(y_test, y_pred)
        }

def evaluate_synthetic_data(real_df, target, epsilons, synth_size=1000, dataset_name=""):
    X_real = real_df.drop(columns=[target])
    y_real = real_df[target]
    classification = is_classification(y_real)

    if classification and y_real.dtype == 'object':
        y_real = y_real.astype('category').cat.codes

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )

    results_by_epsilon = []

    for epsilon in epsilons:
        print(f"\n--- ε = {epsilon} ---")
        #synth = Synthesizer(
         #       synthesizer=PATEGAN(
         #           epsilon=epsilon,
         #           delta=1e-5,          # Small delta for (ε, δ)-differential privacy
         #           binary=True,         # Set True if your target is binary
         #           latent_dim=64,
         #           batch_size=64,
         #           teacher_iters=5,
         #           student_iters=5
         #           ),
         #       epochs=1,                # You can increase epochs for better quality
         #       verbose=True
         #       )
        synth = Synthesizer.create("pategan", epsilon=epsilon, delta=1e-5, binary=True, latent_dim=64, batch_size=64, teacher_iters=10, student_iters=5)
        synth.fit(real_df, preprocessor_eps=1.0)
        filename = f"pkl/pategan/pategan_{dataset_name}_epsilon_{epsilon}.pkl"
        joblib.dump(synth, filename)
        synth_data = synth.sample(synth_size)

        if target not in synth_data.columns:
            print(f"Warning: Target column {target} not found in synthetic data.")
            continue

        X_synth = synth_data.drop(columns=[target])
        y_synth = synth_data[target]

        if classification and y_synth.dtype == 'object':
            y_synth = y_synth.astype('category').cat.codes

        metrics = train_model(X_synth, y_synth, X_real_test, y_real_test, classification)
        results_by_epsilon.append((epsilon, metrics))

        print("Utility (Train on synthetic, test on real):")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: Not Applicable")

    plot_results(results_by_epsilon, classification, dataset_name)

def plot_results(results, classification, dataset_name):
    epsilons = [r[0] for r in results]
    metrics = results[0][1].keys()

    for metric in metrics:
        values = [r[1][metric] for r in results]
        plt.plot(epsilons, values, marker='o', label=metric)

    plt.xlabel("Privacy (ε)")
    plt.ylabel("Utility Score")
    plt.title(f"Privacy-Utility Tradeoff - {dataset_name.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
   # plt.show()
    # Save plot to PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PATEGAN_privacy_utility_{dataset_name}_{timestamp}.pdf"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")


# Set epsilon values
eps_values = [2.0, 3.0, 5.0,7.0,10.0]

# Run for both datasets
for dataset_name in ["adult", "credit"]:
    print(f"\n=== Dataset: {dataset_name.upper()} ===")
    df, target_col = load_dataset(dataset_name)
    print("Target distribution:\n", df[target_col].value_counts())
    evaluate_synthetic_data(df, target_col, eps_values, synth_size=1000, dataset_name=dataset_name)

