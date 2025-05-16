import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from snsynth import Synthesizer
#from snsynth.preprocessors.base import GeneralPurposePreprocessor

def load_dataset(name):
    if name == "adult":
        data = pd.read_csv("../../Real_Datasets/Adult_dataset.csv")  # Ensure preprocessed version
        target = 'income'
    elif name == "credit":
        data = pd.read_csv("../../Real_Datasets/credit_risk_dataset.csv")
        target = 'loan_status'
    else:
        raise ValueError("Unknown dataset")
    data = pd.get_dummies(data, drop_first=True)
 
    return data, target

def train_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred), average_precision_score(y_test, y_pred)

def evaluate_synthetic_data(real_df, target, epsilon=3.0, synth_size=1000):
    # Fit DPCTGAN
    synth = Synthesizer.create("dpctgan", epsilon=epsilon,verbose=True)
    synth.fit(real_df, preprocessor_eps=1.0)
    # Save model to disk
    #synth.save("dpctgan_model.pkl")
    joblib.dump(synth,"dpctgan_model.pkl")
    synth_data = synth.sample(synth_size)


    # Ensure target is in the synthetic data
    if target not in synth_data.columns:
        print(f"Warning: Target column {target} not found in synthetic data.")
        return None

    # Train on synthetic, test on real
    X_real = real_df.drop(columns=[target])
    y_real = real_df[target]
    X_synth = synth_data.drop(columns=[target])
    y_synth = synth_data[target]

    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

    # Train on synthetic, test on real
    auc, auprc = train_classifier(X_synth, y_synth, X_real_test, y_real_test)

    print(f"\nPrivacy ε: {epsilon}")
    print(f"Utility (Train on synthetic, test on real):")
    print(f"  AUROC: {auc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")

    return auc, auprc

# Evaluate for both datasets
for dataset_name in ["adult", "credit"]:
    print(f"\n=== Dataset: {dataset_name.upper()} ===")
    df, target_col = load_dataset(dataset_name)
    evaluate_synthetic_data(df, target_col, epsilon=3.0, synth_size=1000)

