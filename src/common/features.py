import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from src.common.logger import get_logger

logger = get_logger(__name__)


USER_CATEGORICAL_FEATURES = ['TownName', 'GroupHeaderName', 'Cluster', 'RegionCategory', 'Area']
ITEM_CATEGORICAL_FEATURES = ['ProductGroupHeader', 'ProductGroupName']


USER_NUMERIC_FEATURES = ['TenureYears'] 
ITEM_NUMERIC_FEATURES = ['Price']


USER_ID = 'CustomerCode'
ITEM_ID = 'ProductCode'

class Preprocessor:
    def __init__(self, vocab_dir="data/processed/vocabularies"):
        self.vocab_dir = vocab_dir
        os.makedirs(self.vocab_dir, exist_ok=True)
        self.vocabs = {} 
        self.stats = {} 

    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame):
        logger.info("Fitting preprocessor (Generating vocabularies)...")

        for feature in USER_CATEGORICAL_FEATURES + [USER_ID]:
            unique_vals = users_df[feature].dropna().unique().astype(str)

            vocab = {val: i + 1 for i, val in enumerate(unique_vals)}
            self.vocabs[feature] = vocab
            self._save_vocab(feature, unique_vals) 
            
        for feature in ITEM_CATEGORICAL_FEATURES + [ITEM_ID]:
            unique_vals = items_df[feature].dropna().unique().astype(str)
            vocab = {val: i + 1 for i, val in enumerate(unique_vals)}
            self.vocabs[feature] = vocab
            self._save_vocab(feature, unique_vals)

        for feature in USER_NUMERIC_FEATURES:
            self.stats[feature] = {
                'min': float(users_df[feature].min()),
                'max': float(users_df[feature].max())
            }
            
        for feature in ITEM_NUMERIC_FEATURES:
            self.stats[feature] = {
                'min': float(items_df[feature].min()),
                'max': float(items_df[feature].max())
            }
        

        with open(os.path.join(self.vocab_dir, "stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
            
        logger.info("Preprocessor fit complete.")

    def _save_vocab(self, feature_name, unique_values):
        path = os.path.join(self.vocab_dir, f"{feature_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for val in unique_values:
                f.write(f"{val}\n")


    def transform_users(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in USER_CATEGORICAL_FEATURES + [USER_ID]:
            vocab = self.vocabs.get(feature, {})
            df[feature] = df[feature].astype(str).map(vocab).fillna(0).astype(int)
            
        for feature in USER_NUMERIC_FEATURES:
            min_val = self.stats[feature]['min']
            max_val = self.stats[feature]['max']
            denom = (max_val - min_val) if (max_val - min_val) > 0 else 1
            df[feature] = (df[feature] - min_val) / denom
            df[feature] = df[feature].fillna(0.0).astype(float)
            
        return df

    def transform_items(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in ITEM_CATEGORICAL_FEATURES + [ITEM_ID]:
            vocab = self.vocabs.get(feature, {})
            df[feature] = df[feature].astype(str).map(vocab).fillna(0).astype(int)
            
        for feature in ITEM_NUMERIC_FEATURES:
            min_val = self.stats[feature]['min']
            max_val = self.stats[feature]['max']
            denom = (max_val - min_val) if (max_val - min_val) > 0 else 1
            df[feature] = (df[feature] - min_val) / denom
            df[feature] = df[feature].fillna(0.0).astype(float)
            
        return df
