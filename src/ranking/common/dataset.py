import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from src.common.logger import get_logger

logger = get_logger(__name__)

def load_ranking_data(
    train_interactions_path: str,
    test_interactions_path: str,
    users_path: str,
    items_path: str,
    num_negatives_per_positive: int = 5,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:

    np.random.seed(random_seed)
    random.seed(random_seed)

    logger.info("Loading base Parquet files...")

    users_df = pd.read_parquet(users_path)
    items_df = pd.read_parquet(items_path)
    all_item_ids = items_df['ProductCode'].unique()

    train_positives = pd.read_parquet(train_interactions_path)
    test_positives = pd.read_parquet(test_interactions_path)

    train_positives['label'] = 1.0
    test_positives['label'] = 1.0

    logger.info(f"Generating {num_negatives_per_positive} negative samples per positive for TRAINING data...")
    
    all_positives_set = set(
        zip(
            pd.concat([train_positives, test_positives])['CustomerCode'],
            pd.concat([train_positives, test_positives])['ProductCode']
        )
    )

    negative_rows = []
    train_users = train_positives['CustomerCode'].unique()
    
    for user_id in tqdm(train_users, desc="Sampling negatives"):
        n_pos_for_user = len(train_positives[train_positives['CustomerCode'] == user_id])
        n_neg_to_sample = n_pos_for_user * num_negatives_per_positive
        
        candidates = np.random.choice(all_item_ids, size=n_neg_to_sample * 2, replace=True)
        
        valid_negatives = []
        for item_id in candidates:
            if (user_id, item_id) not in all_positives_set:
                valid_negatives.append({'CustomerCode': user_id, 'ProductCode': item_id, 'label': 0.0})
                if len(valid_negatives) >= n_neg_to_sample:
                    break
        
        negative_rows.extend(valid_negatives)

    train_negatives = pd.DataFrame(negative_rows)
    logger.info(f"Generated {len(train_negatives)} negative training samples.")
    logger.info("Combining positives and negatives and joining features...")

    train_final = pd.concat([train_positives, train_negatives], ignore_index=True)
    train_final = train_final.sample(frac=1, random_state=random_seed).reset_index(drop=True)


    test_final = test_positives.copy()
    def join_features(interactions_df):
        df = interactions_df.merge(users_df, on='CustomerCode', how='left')
        df = df.merge(items_df, on='ProductCode', how='left')
        return df.fillna(0)

    train_df = join_features(train_final)
    test_df = join_features(test_final)

    cols_to_drop = ['Recency', 'Frequency', 'Amount'] 
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

    logger.info(f"Final Training Shape: {train_df.shape} (Label mean: {train_df['label'].mean():.2f})")
    logger.info(f"Final Test Shape: {test_df.shape}")

    return train_df, test_df
