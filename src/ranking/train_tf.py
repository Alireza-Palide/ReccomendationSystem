import os
import sys
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ranking.common.dataset import load_ranking_data
from src.ranking.tf_models.deepfm import DeepFM
from src.common.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def df_to_tfdataset(df, batch_size, target_col='label'):
    df = df.copy()
    labels = df.pop(target_col)

    input_dict = {name: tf.constant(value) for name, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((input_dict, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_feature_specs(vocab_dir, categorical_features):
    specs = {}
    for feat in categorical_features:
        path = os.path.join(vocab_dir, f"{feat}.txt")
        try:
            vocab_size = sum(1 for _ in open(path, encoding="utf-8"))
            specs[feat] = vocab_size
        except FileNotFoundError:
             logger.warning(f"Vocab file for {feat} not found. Using default size 10.")
             specs[feat] = 10
    return specs

def train_deepfm():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    model_version = params['serving']['model_version']
    model_dir = os.path.join("models", "ranking", "deepfm_" + model_version)
    os.makedirs(model_dir, exist_ok=True)
    
    ranking_params = params['ranking']
    batch_size = params['data']['batch_size']

    logger.info("Loading data for DeepFM...")
    train_df_full, _ = load_ranking_data(
        params['data']['train_interactions_path'],
        params['data']['test_interactions_path'],
        params['data']['users_path'],
        params['data']['items_path'],
        num_negatives_per_positive=4,
        random_seed=params['random_seed']
    )
    
    train_df_full = train_df_full.drop(columns=['ProductName'], errors='ignore')

    train_df, val_df = train_test_split(
        train_df_full, test_size=0.2, random_state=params['random_seed'], stratify=train_df_full['label']
    )

    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_ds = df_to_tfdataset(train_df, batch_size)
    val_ds = df_to_tfdataset(val_df, batch_size)

    categorical_feats = [
        'CustomerCode', 'ProductCode', 'TownName', 'Cluster', 
        'GroupHeaderName', 'Area', 'RegionCategory',
        'ProductGroupHeader', 'ProductGroupName'
    ]
    
    feature_specs = get_feature_specs(params['data']['vocab_path'], categorical_feats)
    logger.info(f"Feature specs for embeddings: {feature_specs}")

    logger.info("Initializing DeepFM model...")
    model = DeepFM(
        feature_specs=feature_specs,
        embedding_dim=ranking_params['embedding_dim'],
        dnn_layers=ranking_params['layer_sizes'],
        dropout_rate=ranking_params['dropout_rate']
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ranking_params['learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    logger.info("Starting training...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=3, mode='max', restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=1, min_lr=0.0001, verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ranking_params['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    weights_path = os.path.join(model_dir, "deepfm_weights")
    model.save_weights(weights_path)
    logger.info(f"DeepFM weights saved to {weights_path}")

    history_path = os.path.join(model_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    logger.info(f"Training history saved to {history_path}")

    final_val_auc = history.history['val_auc'][-1]
    logger.info(f"Final Validation AUC: {final_val_auc:.4f}")


if __name__ == "__main__":
    train_deepfm()
