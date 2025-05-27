import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, mean_absolute_percentage_error, r2_score,mean_squared_error, explained_variance_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score,mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.multiclass import type_of_target


class SyntheticDataMetrics:
    """
    A utility class to compute metrics for comparing real and synthetic tabular data.
    Metrics include Wasserstein Distance (continuous) and Jensen-Shannon Divergence (categorical).
    """

    @staticmethod
    def calculate_wasserstein(real_data: pd.DataFrame, 
                             synth_data: pd.DataFrame, 
                             num_cols: list) -> float:
        """
        Compute the average Wasserstein Distance (WD) for continuous columns.
        Normalizes data to [0, 1] to ensure fair comparison across features.
        """
        scaler = MinMaxScaler()
        wd_scores = []
        
        for col in num_cols:
            if real_data[col].std() == 0 or synth_data[col].std() == 0:
                print(f"Skipping {col} (zero variance or invalid)")
                continue
                
            real_col = real_data[col].dropna().values.reshape(-1, 1)
            synth_col = synth_data[col].dropna().values.reshape(-1, 1)
            
            real_norm = scaler.fit_transform(real_col).flatten()
            synth_norm = scaler.transform(synth_col).flatten()
            
            wd = wasserstein_distance(real_norm, synth_norm)
            wd_scores.append(wd)
        
        return np.mean(wd_scores) if wd_scores else np.nan

    @staticmethod
    def calculate_jsd(real_data: pd.DataFrame, 
                      synth_data: pd.DataFrame, 
                      cat_cols: list) -> float:
        """
        Compute the average Jensen-Shannon Divergence (JSD) for categorical columns.
        """
        jsd_scores = []
        missing_cats = {}
        
        for col in cat_cols:
            real_counts = real_data[col].value_counts(normalize=True)
            synth_counts = synth_data[col].value_counts(normalize=True)
            
            missing = set(real_counts.index) - set(synth_counts.index)
            if missing:
                missing_cats[col] = list(missing)
            
            all_cats = real_counts.index.union(synth_counts.index)
            real_probs = real_counts.reindex(all_cats, fill_value=0).values
            synth_probs = synth_counts.reindex(all_cats, fill_value=0).values
            
            jsd = jensenshannon(real_probs, synth_probs, base=2) ** 2
            jsd_scores.append(jsd)
        
        if missing_cats:
            print(f"Warning: Missing categories detected: {missing_cats}")
        
        return np.mean(jsd_scores) if jsd_scores else np.nan

    @staticmethod
    def compare(real_data: pd.DataFrame, 
               synth_data: pd.DataFrame, 
               num_cols: list, 
               cat_cols: list) -> dict:
        """
        Compare real and synthetic data using WD (continuous) and JSD (categorical).
        """
        results = {}
        
        if num_cols:
            results['wasserstein'] = SyntheticDataMetrics.calculate_wasserstein(
                real_data, synth_data, num_cols
            )
        
        if cat_cols:
            results['jsd'] = SyntheticDataMetrics.calculate_jsd(
                real_data, synth_data, cat_cols
            )
        
        return results

    @staticmethod
    def is_classification(y):
        """More robust classification detection"""
        if len(y) == 0:
            return True  # Default to classification for empty data
        if pd.api.types.is_categorical_dtype(y):
            return True
        if pd.api.types.is_object_dtype(y):
            return True
        unique_values = len(np.unique(y.dropna()))
        return unique_values < min(20, len(y)/10)  # Dynamic threshold

    @staticmethod
    def model_inversion_attack(real_df, synth_df, target_col):
        """
        Robust model inversion attack with proper label and feature handling
        """
        X_synth = synth_df.drop(columns=[target_col])
        y_synth = synth_df[target_col]
        X_real = real_df.drop(columns=[target_col])
        y_real = real_df[target_col]

        # Encode categorical features in X_synth and X_real
        cat_cols = X_synth.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            # Ensure consistent types and handle missing values
            X_synth[cat_cols] = X_synth[cat_cols].astype(str).fillna("NA")
            X_real[cat_cols] = X_real[cat_cols].astype(str).fillna("NA")

            # Fit encoder on combined data to ensure shared categories
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            enc.fit(pd.concat([X_synth[cat_cols], X_real[cat_cols]], axis=0))

            X_synth[cat_cols] = enc.transform(X_synth[cat_cols])
            X_real[cat_cols] = enc.transform(X_real[cat_cols])

        # Determine problem type
        try:
            problem_type = type_of_target(y_real)
            classification = problem_type in ['binary', 'multiclass']
        except ValueError:
            classification = len(y_real.unique()) / len(y_real) < 0.05

        if classification:
            # Convert labels to strings for safe comparison
            y_synth = y_synth.astype(str)
            y_real = y_real.astype(str)

            # Find common labels
            common_labels = set(y_synth.unique()).intersection(set(y_real.unique()))

            if len(common_labels) < 2:
                return {
                    'Model inversion attack': 'Failed - Insufficient common classes',
                    'Real classes': y_real.value_counts().to_dict(),
                    'Synthetic classes': y_synth.value_counts().to_dict(),
                    'Common classes': list(common_labels)
                }

            # Filter to only common label samples
            synth_mask = y_synth.isin(common_labels)
            real_mask = y_real.isin(common_labels)

            if sum(synth_mask) == 0 or sum(real_mask) == 0:
                return {'Model inversion attack': 'Failed - No matching classes'}

            le = LabelEncoder().fit(list(common_labels))

            try:
                clf = RandomForestClassifier(random_state=0)
                clf.fit(X_synth[synth_mask], le.transform(y_synth[synth_mask]))

                y_pred = le.inverse_transform(clf.predict(X_real[real_mask]))
                acc = accuracy_score(y_real[real_mask], y_pred)

                if len(common_labels) == 2:
                    y_proba = clf.predict_proba(X_real[real_mask])[:, 1]
                    auroc = roc_auc_score(y_real[real_mask], y_proba)
                else:
                    auroc = None

                return {
                    'Accuracy': float(acc) if acc is not None else None,
                    'AUROC': float(auroc) if auroc is not None else None,
                    'Classes_used': str(common_labels)
                }
            except Exception as e:
                return {'Model inversion attack': f'Failed: {str(e)}'}
        else:
            # Regression case: ensure numeric labels
            y_synth = pd.to_numeric(y_synth, errors='coerce')
            y_real = pd.to_numeric(y_real, errors='coerce')

            valid_mask_synth = ~y_synth.isna()
            valid_mask_real = ~y_real.isna()

            if valid_mask_synth.sum() == 0 or valid_mask_real.sum() == 0:
                return {'Model inversion attack': 'Failed - Invalid numeric values'}

            try:
                reg = RandomForestRegressor(random_state=0)
                reg.fit(X_synth[valid_mask_synth], y_synth[valid_mask_synth])
                y_pred = reg.predict(X_real[valid_mask_real])

                return {
                    'MSE': mean_squared_error(y_real[valid_mask_real], y_pred),
                    'R2': r2_score(y_real[valid_mask_real], y_pred)
                }
            except Exception as e:
                return {'Model inversion attack': f'Regression failed: {str(e)}'}


    @staticmethod
    def membership_inference_attack(real_df, synth_df, target_col):
        # Drop target column (we only care about feature leakage)
        X_real = real_df.drop(columns=[target_col])
        X_synth = synth_df.drop(columns=[target_col])

        # Label real=1, synthetic=0
        X = pd.concat([X_real, X_synth], axis=0)
        y = np.array([1] * len(X_real) + [0] * len(X_synth))

        # Encode categorical variables properly
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes

        # Split into train/test to avoid overfitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train on train set, evaluate on test set
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_proba)

        return {"MIA_AUROC": auroc}
    
    @staticmethod
    def pearson_correlation_difference(real_df, synth_df, continuous_cols):
        """
        Calculate mean absolute difference in Pearson correlations for CONTINUOUS columns only.
        
        Args:
            real_df: DataFrame of real data
            synth_df: DataFrame of synthetic data
            continuous_cols: List of continuous column names
            
        Returns:
            float: Mean absolute difference in correlation coefficients
        """
        common_cont = [col for col in continuous_cols if col in real_df.columns and col in synth_df.columns]
        if not common_cont:
            return np.nan
            
        real_corr = real_df[common_cont].corr(method='pearson').values
        synth_corr = synth_df[common_cont].corr(method='pearson').values
        
        # Only consider upper triangle without diagonal
        mask = np.triu_indices_from(real_corr, k=1)
        diff = np.abs(real_corr[mask] - synth_corr[mask])
        return np.mean(diff)

    def uncertainty_coefficient(x, y):
        from sklearn.metrics import mutual_info_score
        h_x = mutual_info_score(x, x)
        if h_x == 0:
            return 0
        return mutual_info_score(x, y) / h_x
    
    def correlation_ratio(categories, values):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg = np.mean(values)
        numerator = 0
        for i in range(cat_num):
            # Use iloc for positional indexing
            cat_values = values.iloc[np.argwhere(fcat == i).flatten()]
            if len(cat_values) > 0:
                numerator += len(cat_values) * (np.mean(cat_values) - y_avg) ** 2
        denominator = np.sum((values - y_avg) ** 2)
        return np.sqrt(numerator / denominator) if denominator != 0 else 0


    @staticmethod
    def uncertainty_coefficient_difference(real_df, synth_df, categorical_cols):
        """
        Calculate mean absolute difference in uncertainty coefficients for CATEGORICAL columns only.
        
        Args:
            real_df: DataFrame of real data
            synth_df: DataFrame of synthetic data  
            categorical_cols: List of categorical column names
            
        Returns:
            float: Mean absolute difference in uncertainty coefficients
        """
        common_cat = [col for col in categorical_cols if col in real_df.columns and col in synth_df.columns]
        if not common_cat:
            return np.nan
            
        # Encode categorical variables
        real_enc = real_df[common_cat].apply(LabelEncoder().fit_transform)
        synth_enc = synth_df[common_cat].apply(LabelEncoder().fit_transform)
        
        uc_diffs = []
        for i, col1 in enumerate(common_cat):
            for col2 in common_cat[i+1:]:  # Avoid duplicate pairs
                real_uc = SyntheticDataMetrics.uncertainty_coefficient(real_enc[col1], real_enc[col2])
                synth_uc = SyntheticDataMetrics.uncertainty_coefficient(synth_enc[col1], synth_enc[col2])
                uc_diffs.append(abs(real_uc - synth_uc))
        
        return np.mean(uc_diffs) if uc_diffs else 0

    @staticmethod 
    def correlation_ratio_difference(real_df, synth_df, categorical_cols, continuous_cols):
        """
        Calculate mean absolute difference in correlation ratios between:
        - categorical (X) and continuous (Y) columns
        - continuous (X) and categorical (Y) columns
        
        Args:
            real_df: DataFrame of real data
            synth_df: DataFrame of synthetic data
            categorical_cols: List of categorical column names
            continuous_cols: List of continuous column names
            
        Returns:
            float: Mean absolute difference in correlation ratios
        """
        common_cat = [col for col in categorical_cols if col in real_df.columns and col in synth_df.columns]
        common_cont = [col for col in continuous_cols if col in real_df.columns and col in synth_df.columns]
        
        if not common_cat or not common_cont:
            return np.nan
            
        # Encode categorical variables
        real_cat = real_df[common_cat].apply(LabelEncoder().fit_transform)
        synth_cat = synth_df[common_cat].apply(LabelEncoder().fit_transform)
        real_cont = real_df[common_cont]
        synth_cont = synth_df[common_cont]
        
        cr_diffs = []
        
        # Categorical (X) vs Continuous (Y)
        for cat_col in common_cat:
            for cont_col in common_cont:
                real_cr = SyntheticDataMetrics.correlation_ratio(real_cat[cat_col], real_cont[cont_col])
                synth_cr = SyntheticDataMetrics.correlation_ratio(synth_cat[cat_col], synth_cont[cont_col])
                cr_diffs.append(abs(real_cr - synth_cr))
        
        # Continuous (X) vs Categorical (Y) - different relationship
        for cont_col in common_cont:
            for cat_col in common_cat:
                real_cr = SyntheticDataMetrics.correlation_ratio(real_cont[cont_col], real_cat[cat_col])
                synth_cr = SyntheticDataMetrics.correlation_ratio(synth_cont[cont_col], synth_cat[cat_col])
                cr_diffs.append(abs(real_cr - synth_cr))
        
        return np.mean(cr_diffs) if cr_diffs else 0