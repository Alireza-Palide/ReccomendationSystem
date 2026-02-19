import tensorflow as tf
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple

def load_dataset(
    interactions_path: str,
    users_path: str,
    items_path: str,
    batch_size: int = 2048
) -> tf.data.Dataset:

    interactions = pd.read_parquet(interactions_path)
    users = pd.read_parquet(users_path)
    items = pd.read_parquet(items_path)
    data = interactions.merge(users, on="CustomerCode", how="left")
    data = data.merge(items, on="ProductCode", how="left")
    

    inputs = {
        "CustomerCode": data["CustomerCode"].values.astype(np.int32),
        "ProductCode": data["ProductCode"].values.astype(np.int32),
        "TownName": data["TownName"].values.astype(np.int32),
        "Cluster": data["Cluster"].values.astype(np.int32),
        "GroupHeaderName": data["GroupHeaderName"].values.astype(np.int32),
        "Area": data["Area"].values.astype(np.int32),
        "RegionCategory": data["RegionCategory"].values.astype(np.int32),
        "TenureYears": data["TenureYears"].values.astype(np.float32),
        "ProductGroupHeader": data["ProductGroupHeader"].values.astype(np.int32),
        "ProductGroupName": data["ProductGroupName"].values.astype(np.int32),
        "Price": data["Price"].values.astype(np.float32),
        "IsBestSeller": data["IsBestSeller"].values.astype(np.float32),
    }
    

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    dataset = dataset.shuffle(buffer_size=100_000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_candidates_dataset(items_path: str, batch_size: int = 128) -> tf.data.Dataset:
    items = pd.read_parquet(items_path)
    
    inputs = {
        "ProductCode": items["ProductCode"].values.astype(np.int32),
        "ProductGroupHeader": items["ProductGroupHeader"].values.astype(np.int32),
        "ProductGroupName": items["ProductGroupName"].values.astype(np.int32),
        "Price": items["Price"].values.astype(np.float32),
        "IsBestSeller": items["IsBestSeller"].values.astype(np.float32),
    }
    
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

