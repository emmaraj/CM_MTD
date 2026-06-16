"""
CICIDS2017 Dataset Loader for CM-MTD Framework.

The CICIDS-2017 dataset contains 5 days of network traffic:
- Monday    : BENIGN
- Tuesday   : FTP-Patator, SSH-Patator (BruteForce)
- Wednesday : DoS (Hulk, GoldenEye, Slowhttptest, Slowloris), Heartbleed
- Thursday  : WebAttacks (BruteForce, XSS, SQLInjection), Infiltration
- Friday    : Bot, PortScan, DDoS

Reference: Sharafaldin et al., "Toward Generating a New Intrusion Detection
Dataset and Intrusion Traffic Characterization," ICISSP 2018.

Download from: https://www.unb.ca/cic/datasets/ids-2017.html
Place the 8 CSV files in data/cicids2017/
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.compat import tqdm  # graceful fallback if tqdm not installed

logger = logging.getLogger("cm_mtd.data")

# ─── Known column name variants in CICIDS2017 CSVs ─────────────────────────
# The original files have leading spaces in column names.
_LABEL_COLUMN_VARIANTS = [
    "Label", " Label", "label", " label",
    "Label ", " Label ",
]

# Columns to drop (identifiers, not features)
_DROP_COLUMNS = [
    "Flow ID", " Flow ID", "Source IP", " Source IP",
    "Source Port", " Source Port", "Destination IP", " Destination IP",
    "Destination Port", " Destination Port", "Protocol", " Protocol",
    "Timestamp", " Timestamp",
]

# CICIDS2017 raw label → unified attack class
_RAW_LABEL_MAP = {
    "BENIGN": "BENIGN",
    "Benign": "BENIGN",
    "benign": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack Brute Force": "WebAttack",
    "Web Attack \x96 XSS": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack XSS": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Web Attack Sql Injection": "WebAttack",
}

# Unified class → integer encoding
ATTACK_CLASS_MAP = {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    "PortScan": 3,
    "Infiltration": 4,
    "Bot": 5,
    "BruteForce": 6,
    "WebAttack": 7,
}

CLASS_NAMES = list(ATTACK_CLASS_MAP.keys())
N_CLASSES = len(ATTACK_CLASS_MAP)


class CICIDS2017Loader:
    """
    Loads and merges all CICIDS2017 CSV files into a unified DataFrame.
    
    Usage:
        loader = CICIDS2017Loader(data_dir="data/cicids2017")
        df = loader.load_all()
    """

    def __init__(
        self,
        data_dir: str = "data/cicids2017",
        label_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.label_map = label_map or _RAW_LABEL_MAP

    def load_all(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load and merge all CSV files in the data directory.
        
        Args:
            verbose: Print progress information.
        
        Returns:
            Merged DataFrame with unified label column.
        
        Raises:
            FileNotFoundError: If no CSV files found and synthetic fallback disabled.
        """
        csv_files = sorted(self.data_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(
                f"No CSV files found in {self.data_dir}. "
                "Please download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
                "and place the CSV files in data/cicids2017/"
            )
            raise FileNotFoundError(
                f"No CICIDS2017 CSV files found in {self.data_dir}"
            )

        if verbose:
            logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")

        dfs = []
        for csv_path in tqdm(csv_files, desc="Loading CICIDS2017 CSVs", disable=not verbose):
            df = self._load_single_file(csv_path)
            if df is not None:
                dfs.append(df)
                if verbose:
                    logger.info(
                        f"  {csv_path.name}: {len(df):,} rows | "
                        f"Labels: {df['label'].value_counts().to_dict()}"
                    )

        if not dfs:
            raise ValueError("All CSV files failed to load.")

        merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total merged rows: {len(merged):,}")
        logger.info(f"Class distribution:\n{merged['label'].value_counts()}")

        return merged

    def _load_single_file(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single CICIDS2017 CSV file.
        
        Handles:
        - Leading/trailing whitespace in column names
        - Encoding issues (latin-1 fallback)
        - Missing or NaN label column
        """
        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

            # Strip whitespace from column names
            df.columns = [str(c).strip() for c in df.columns]

            # Find label column
            label_col = None
            for variant in _LABEL_COLUMN_VARIANTS:
                stripped = variant.strip()
                if stripped in df.columns:
                    label_col = stripped
                    break

            if label_col is None:
                logger.warning(f"No label column found in {csv_path.name}. Columns: {df.columns.tolist()}")
                return None

            # Rename to standardized 'label'
            df = df.rename(columns={label_col: "label"})

            # Strip whitespace from label values and map
            df["label"] = df["label"].astype(str).str.strip()
            df["label"] = df["label"].map(self.label_map)

            # Drop rows with unknown labels
            unknown_mask = df["label"].isna()
            if unknown_mask.sum() > 0:
                logger.debug(f"  Dropped {unknown_mask.sum()} rows with unknown labels in {csv_path.name}")
            df = df.dropna(subset=["label"])

            # Drop non-feature columns
            drop_cols = [c for c in _DROP_COLUMNS if c.strip() in df.columns]
            df = df.drop(columns=drop_cols, errors="ignore")

            # Add source filename (for traceability)
            df["_source_file"] = csv_path.name

            return df

        except Exception as e:
            logger.error(f"Failed to load {csv_path.name}: {e}")
            return None

    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Return class distribution as a dictionary."""
        return df["label"].value_counts().to_dict()
