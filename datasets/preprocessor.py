"""
Preprocessing Pipeline for CICIDS2017 Dataset.

Steps:
1. Remove duplicate flows
2. Handle missing / infinite values
3. Encode attack labels (integer + one-hot)
4. Feature selection and normalization
5. Class balancing (SMOTE or undersampling)
6. Train/validation/test split

This module produces clean feature matrices and label vectors ready
for LSTM sequence construction and baseline ML models.
"""
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")
logger = logging.getLogger("cm_mtd.preprocessor")

from datasets.cicids2017_loader import ATTACK_CLASS_MAP, CLASS_NAMES, N_CLASSES


class CICIDS2017Preprocessor:
    """
    Full preprocessing pipeline for CICIDS2017.
    
    Args:
        config: Data configuration dictionary from config.yaml.
    """

    # Features to exclude beyond the standard drop list
    _EXCLUDE_FEATURES = {"label", "_source_file"}

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    # ─── Main Pipeline ──────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit the preprocessor on data and transform it.
        
        Args:
            df: Raw merged DataFrame from CICIDS2017Loader.
        
        Returns:
            Tuple of (X: features, y: integer labels, feature_names).
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 2: Handle inf / NaN values
        df = self._handle_missing(df)

        # Step 3: Extract features and labels
        X, y = self._extract_features_labels(df)

        # Step 4: Fit scaler on training data
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        logger.info(f"Preprocessing complete: X={X_scaled.shape}, y={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        return X_scaled, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using the fitted scaler.
        
        Args:
            df: Raw DataFrame (must have same columns as training data).
        
        Returns:
            Tuple of (X_scaled, y).
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df = self._remove_duplicates(df)
        df = self._handle_missing(df)
        X, y = self._extract_features_labels(df)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    # ─── Pipeline Steps ─────────────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows (same flow features and label)."""
        before = len(df)
        # Use all columns except metadata for dedup
        dedup_cols = [c for c in df.columns if c not in {"_source_file"}]
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        after = len(df)
        logger.info(f"  Duplicates removed: {before - after:,} ({100*(before-after)/max(before,1):.1f}%)")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/NaN with column medians, then drop any remaining NaN rows."""
        # Replace ±infinity
        df = df.replace([np.inf, -np.inf], np.nan)

        # Count NaN before
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.info(f"  NaN values found: {nan_count:,}. Filling with column medians.")
            # Fill NaN with median for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                median_val = df[col].median()
                if np.isnan(median_val):
                    median_val = 0.0
                df[col].fillna(median_val, inplace=True)

        # Drop any remaining NaN rows (e.g. label column)
        df = df.dropna()
        return df

    def _extract_features_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract numeric features and encode labels as integers."""
        # Identify feature columns (numeric, not metadata)
        exclude = self._EXCLUDE_FEATURES
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(np.float32)

        # Encode labels
        label_series = df["label"]
        y = np.array([ATTACK_CLASS_MAP.get(lbl, 0) for lbl in label_series], dtype=np.int64)

        return X, y

    # ─── Splitting ──────────────────────────────────────────────────────────

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        stratify: bool = True,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train / validation / test sets.
        
        Args:
            X: Feature matrix.
            y: Label array.
            train_frac: Fraction for training.
            val_frac: Fraction for validation.
            test_frac: Fraction for test.
            stratify: Use stratified splitting.
            seed: Random seed.
        
        Returns:
            Dict with keys 'train', 'val', 'test' and tuple values (X, y).
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
            "Fractions must sum to 1.0"

        stratify_arg = y if stratify else None

        # First: split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_frac,
            stratify=stratify_arg,
            random_state=seed,
        )

        # Then split remaining into train + val
        val_relative = val_frac / (train_frac + val_frac)
        stratify_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            stratify=stratify_temp,
            random_state=seed,
        )

        logger.info(
            f"Split sizes — Train: {len(X_train):,} | "
            f"Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        return {
            "train": (X_train, y_train),
            "val":   (X_val,   y_val),
            "test":  (X_test,  y_test),
        }

    # ─── Class Balancing ────────────────────────────────────────────────────

    def balance_smote(
        self, X: np.ndarray, y: np.ndarray, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to balance minority classes.
        
        Args:
            X: Feature matrix.
            y: Label array.
            seed: Random seed.
        
        Returns:
            Balanced (X, y).
        """
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=seed, k_neighbors=3)
            X_res, y_res = sm.fit_resample(X, y)
            logger.info(f"  SMOTE applied: {len(X):,} → {len(X_res):,} samples")
            return X_res, y_res
        except ImportError:
            logger.warning("imbalanced-learn not available. Skipping SMOTE.")
            return X, y
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}). Falling back to original data.")
            return X, y

    def balance_undersample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_per_class: int = 50000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersample majority classes to max_per_class samples.
        
        Args:
            X: Feature matrix.
            y: Label array.
            max_per_class: Maximum samples per class.
            seed: Random seed.
        
        Returns:
            Balanced (X, y).
        """
        rng = np.random.default_rng(seed)
        indices = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) > max_per_class:
                cls_idx = rng.choice(cls_idx, max_per_class, replace=False)
            indices.extend(cls_idx)

        rng.shuffle(indices)
        return X[indices], y[indices]
