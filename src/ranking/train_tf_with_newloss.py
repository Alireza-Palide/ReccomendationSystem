import os
import sys
import yaml
import pandas as pd
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



class DeepFMWithDiversity(DeepFM):

    def __init__(self, lambda_div=0.01, topk=5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_div = lambda_div
        self.topk = topk
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            bce_loss = self.loss_fn(y, y_pred)
            scores = tf.squeeze(y_pred, axis=1)
            k = tf.minimum(self.topk, tf.shape(scores)[0])
            values, indices = tf.math.top_k(scores, k=k)

            item_ids = tf.gather(x['ProductCode'], indices)
            item_emb_layer = self.embedding_layers['ProductCode']
            item_embs = item_emb_layer(item_ids)

            normed = tf.nn.l2_normalize(item_embs, axis=1)
            sim_matrix = tf.matmul(normed, normed, transpose_b=True)
            y_outer = tf.tensordot(values, values, axes=0)
            diversity_penalty = tf.reduce_sum(y_outer * sim_matrix)

            total_loss = bce_loss + self.lambda_div * diversity_penalty

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            "div_loss": diversity_penalty,
            **{m.name: m.result() for m in self.metrics}
        }



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
        train_df_full,
        test_size=0.2,
        random_state=params['random_seed'],
        stratify=train_df_full['label']
    )

    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_ds = df_to_tfdataset(train_df, batch_size)
    val_ds = df_to_tfdataset(val_df, batch_size)

    categorical_feats = [
        'CustomerCode', 'ProductCode', 'TownName', 'Cluster',
        'GroupHeaderName', 'Area', 'RegionCategory',
        'ProductGroupHeader', 'ProductGroupName'
    ]

    feature_specs = {}
    for feat in categorical_feats:
        vocab_path = os.path.join(params['data']['vocab_path'], f"{feat}.txt")
        try:
            vocab_size = sum(1 for _ in open(vocab_path, encoding="utf-8"))
            feature_specs[feat] = vocab_size
        except FileNotFoundError:
            logger.warning(f"Vocab file for {feat} not found. Using default size 10.")
            feature_specs[feat] = 10

    logger.info(f"Feature specs: {feature_specs}")

    logger.info("Initializing DeepFMWithDiversity model...")

    model = DeepFMWithDiversity(
        feature_specs=feature_specs,
        embedding_dim=ranking_params['embedding_dim'],
        dnn_layers=ranking_params['layer_sizes'],
        dropout_rate=ranking_params['dropout_rate'],
        lambda_div=ranking_params.get('lambda_div', 0.01),
        topk=ranking_params.get('diversity_topk', 5)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=ranking_params['learning_rate']
        ),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=1,
            min_lr=0.0001,
            verbose=1
        )
    ]

    logger.info("Starting training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ranking_params['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    weights_path = os.path.join(model_dir, "deepfm_weights")
    model.save_weights(weights_path)
    logger.info(f"Weights saved to {weights_path}")

    history_path = os.path.join(model_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    logger.info("Training completed.")


if __name__ == "__main__":
    train_deepfm()