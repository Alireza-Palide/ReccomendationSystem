import pandas as pd
import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.features import Preprocessor
from src.common.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def process():

    logger.info("Loading raw data...")
    users_raw = pd.read_parquet("data/raw/customers.parquet")
    items_raw = pd.read_parquet("data/raw/products.parquet")
    interactions_raw = pd.read_parquet("data/raw/interactions.parquet")

    preprocessor = Preprocessor()
    preprocessor.fit(users_raw, items_raw)

    logger.info("Transforming features...")
    users_processed = preprocessor.transform_users(users_raw)
    items_processed = preprocessor.transform_items(items_raw)

    logger.info("Processing interactions...")
    
    user_vocab = preprocessor.vocabs['CustomerCode']
    item_vocab = preprocessor.vocabs['ProductCode']
    
    interactions_processed = interactions_raw.copy()
    interactions_processed['CustomerCode'] = interactions_processed['CustomerCode'].astype(str).map(user_vocab).fillna(0).astype(int)
    interactions_processed['ProductCode'] = interactions_processed['ProductCode'].astype(str).map(item_vocab).fillna(0).astype(int)
    
    before_len = len(interactions_processed)
    interactions_processed = interactions_processed[
        (interactions_processed['CustomerCode'] > 0) & 
        (interactions_processed['ProductCode'] > 0)
    ]
    logger.info(f"Dropped {before_len - len(interactions_processed)} interactions due to unknown IDs.")

    logger.info("Splitting data...")
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(interactions_processed, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    users_processed.to_parquet("data/processed/users_final.parquet", index=False)
    items_processed.to_parquet("data/processed/items_final.parquet", index=False)
    train_df.to_parquet("data/processed/train_interactions.parquet", index=False)
    test_df.to_parquet("data/processed/test_interactions.parquet", index=False)
    
    with open("data/processed/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    logger.info("Data processing complete. Ready for training.")

if __name__ == "__main__":
    process()
