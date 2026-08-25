"""
OFI Predictor: Machine learning model for price movement prediction
Trains logistic regression on order flow imbalance features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
import os

class OFIPredictor:
    """
    Trains and evaluates ML model to predict price movements from OFI features.
    Uses logistic regression with balanced class weights.
    """
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metrics = {}
        
    def load_data(self, csv_path):
        """Load and validate training data"""
        df = pd.read_csv(csv_path)
        
        print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Validate label distribution
        label_counts = df['price_increase'].value_counts()
        label_pct = df['price_increase'].value_counts(normalize=True) * 100
        
        print(f"\nLabel distribution:")
        print(f"  Class 0 (decrease): {label_counts[0]} ({label_pct[0]:.1f}%)")
        print(f"  Class 1 (increase): {label_counts[1]} ({label_pct[1]:.1f}%)")
        
        return df
    
    def prepare_features(self, df):
        """Extract feature matrix and labels"""
        # actual_horizon_seconds is diagnostic metadata from add_labels() (how far
        # ahead the label actually looked) — not something known at prediction time,
        # so it must never be a model input.
        exclude_cols = ['timestamp', 'mid_price', 'future_price', 'actual_horizon_seconds',
                        'source_session', 'price_increase']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df['price_increase'].values
        
        print(f"\nFeatures selected: {len(self.feature_columns)}")
        
        return X, y
    
    def train(self, csv_path, test_size=0.2, handle_imbalance=True, verbose=True):
        """
        Train prediction model with time-based train/test split
        
        Args:
            csv_path: Path to training data CSV
            test_size: Fraction for test set (default: 0.2)
            handle_imbalance: Use balanced class weights
            verbose: Print detailed results
        """
        if verbose:
            print("=" * 80)
            print("TRAINING OFI PREDICTION MODEL")
            print("=" * 80)
        
        # Load and prepare data
        df = self.load_data(csv_path)
        X, y = self.prepare_features(df)
        
        # Time-based split (no shuffling for time series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nTrain/test split: {len(X_train)} / {len(X_test)} samples")
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced' if handle_imbalance else None,
                max_iter=1000,
                random_state=42
            )
        
        print(f"\nTraining {self.model_type} model...")
        if handle_imbalance:
            print("  Using balanced class weights")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, test_pred)
        test_prec = precision_score(y_test, test_pred, zero_division=0)
        test_rec = recall_score(y_test, test_pred, zero_division=0)
        baseline = max(np.sum(y_test == 0), np.sum(y_test == 1)) / len(y_test)
        cm = confusion_matrix(y_test, test_pred)
        
        # Store metrics
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'baseline_accuracy': baseline,
            'confusion_matrix': cm
        }
        
        if verbose:
            self._print_results(y_train, train_pred, y_test, test_pred)
        
        return self.metrics
    
    def _print_results(self, y_train, train_pred, y_test, test_pred):
        """Print training results"""
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        # Training set
        print("\nTraining Set:")
        print(f"  Accuracy:  {accuracy_score(y_train, train_pred):.3f}")
        print(f"  Precision: {precision_score(y_train, train_pred, zero_division=0):.3f}")
        print(f"  Recall:    {recall_score(y_train, train_pred, zero_division=0):.3f}")
        
        # Test set (most important)
        print("\nTest Set:")
        print(f"  Accuracy:  {self.metrics['test_accuracy']:.3f}")
        print(f"  Precision: {self.metrics['test_precision']:.3f}")
        print(f"  Recall:    {self.metrics['test_recall']:.3f}")
        
        # Baseline comparison
        print(f"\nBaseline (majority class): {self.metrics['baseline_accuracy']:.3f}")
        improvement = self.metrics['test_accuracy'] - self.metrics['baseline_accuracy']
        print(f"Improvement over baseline: {improvement:+.3f}")
        
        if improvement > 0:
            print("Status: Model outperforms baseline")
        else:
            print("Status: Model underperforms baseline - more data needed")
        
        # Confusion matrix
        cm = self.metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        
        # Feature importance
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\nTop 5 Features by Importance:")
            for idx, row in importance.head(5).iterrows():
                print(f"  {row['feature']:20s} {row['coefficient']:+.4f}")
    
    def cross_validate(self, csv_path, n_splits=5, handle_imbalance=True, verbose=True):
        """
        Time-series cross-validation (walk-forward, no shuffling — each fold trains
        on the past and tests on the chunk right after it, same as train()'s split
        but repeated 5x instead of once).

        A single 80/20 split on noisy short-horizon financial data is unreliable —
        this project's own numbers proved it: one run of train() gave 57.4% test
        accuracy, a later run gave 54.4%, on what was meant to be the same setup.
        Report the mean +/- std across folds as the honest headline number, not
        whatever a single split happened to land on.
        """
        if verbose:
            print("=" * 80)
            print(f"TIME-SERIES CROSS-VALIDATION ({n_splits} folds)")
            print("=" * 80)

        df = self.load_data(csv_path) if verbose else pd.read_csv(csv_path)
        X, y = self.prepare_features(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(
                class_weight='balanced' if handle_imbalance else None,
                max_iter=1000, random_state=42,
            )
            model.fit(X_train_scaled, y_train)
            test_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, test_pred)
            baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
            fold_metrics.append({
                'fold': i + 1, 'train_size': len(train_idx), 'test_size': len(test_idx),
                'accuracy': acc, 'baseline': baseline, 'edge': acc - baseline,
            })
            if verbose:
                print(f"  Fold {i+1}: train={len(train_idx):4d} test={len(test_idx):4d} "
                      f"| acc={acc:.3f} | baseline={baseline:.3f} | edge={acc - baseline:+.3f}")

        accs = np.array([m['accuracy'] for m in fold_metrics])
        baselines = np.array([m['baseline'] for m in fold_metrics])
        edges = np.array([m['edge'] for m in fold_metrics])

        summary = {
            'fold_metrics': fold_metrics,
            'mean_accuracy': accs.mean(), 'std_accuracy': accs.std(),
            'mean_baseline': baselines.mean(), 'mean_edge': edges.mean(), 'std_edge': edges.std(),
        }

        if verbose:
            print(f"\nMean accuracy:  {summary['mean_accuracy']:.3f} +/- {summary['std_accuracy']:.3f}")
            print(f"Mean baseline:  {summary['mean_baseline']:.3f}")
            print(f"Mean edge:      {summary['mean_edge']:+.3f} +/- {summary['std_edge']:.3f}")
            consistent = (edges > 0).all()
            print(f"Beats baseline in all {n_splits} folds: {consistent}")

        return summary

    def predict(self, features):
        """
        Make prediction for new features
        
        Args:
            features: Dict or DataFrame with feature values
            
        Returns:
            Dict with prediction, probabilities, and confidence
        """
        if isinstance(features, dict):
            X = np.array([[features[col] for col in self.feature_columns]])
        else:
            X = features[self.feature_columns].values
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_decrease': probability[0],
            'probability_increase': probability[1],
            'confidence': max(probability)
        }
    
    def save_model(self, path='models/ofi_predictor.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        print(f"\nModel saved to: {path}")
    
    def load_model(self, path='models/ofi_predictor.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        print(f"Model loaded from: {path}")

if __name__ == "__main__":
    predictor = OFIPredictor(model_type='logistic')

    # Relabeled with the corrected time-based horizon + deadzone logic in
    # FeatureEngineer.add_labels() — run `python relabel_data.py <raw_csv>` on a
    # freshly-collected file first if you're not using this one.
    csv_path = "data/training_data_20251118_152030_relabeled.csv"

    metrics = predictor.train(csv_path, test_size=0.2, handle_imbalance=True)
    predictor.save_model()

    # A single 80/20 split is noisy on data this size — cross-validate before
    # trusting the number above (see cross_validate()'s docstring for why).
    predictor.cross_validate(csv_path, n_splits=5, handle_imbalance=True)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)