import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import sys
import yaml
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.dataset import load_dataset, load_candidates_dataset
from src.retrieval.model import RetrievalModel
from src.common.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    version_dir = os.path.join("models", "retrieval", params['serving']['model_version'])
    os.makedirs(version_dir, exist_ok=True)
    
    logger.info("Loading Training Data...")
    train_ds = load_dataset(
        params['data']['train_interactions_path'], 
        params['data']['users_path'],            
        params['data']['items_path'],         
        batch_size=params['data']['batch_size']
    )
    logger.info("Loading Validation (Test) Data...")
    val_ds = load_dataset(
        params['data']['test_interactions_path'], 
        params['data']['users_path'],
        params['data']['items_path'],
        batch_size=params['data']['batch_size'] // 2 
    )
    
    logger.info("Loading Candidate Pool...")
    candidates_ds = load_candidates_dataset(
        params['data']['items_path']
    )

    logger.info("Initializing Two-Tower Model...")
    model = RetrievalModel(
        layer_sizes=params['retrieval']['layer_sizes'],
        vocab_dir=params['data']['vocab_path'],
        candidates_dataset=candidates_ds
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=params['retrieval']['learning_rate']
    ))

    logger.info("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_freq=1, 
        epochs=params['retrieval']['epochs'],
        verbose=1
    )
    
    history_path = os.path.join(version_dir, "history.pkl")
    logger.info(f"Saving training metrics to {history_path}...")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

    logger.info("Building BruteForce Index...")

    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    
    index.index_from_dataset(
        tf.data.Dataset.zip((
            candidates_ds.map(lambda x: x["ProductCode"]),
            candidates_ds.map(model.candidate_model)
        ))
    )


    dummy_input = {
        "CustomerCode": tf.constant([1], dtype=tf.int32),
        "TownName": tf.constant([1], dtype=tf.int32),
        "Cluster": tf.constant([1], dtype=tf.int32),
        "GroupHeaderName": tf.constant([1], dtype=tf.int32),
        "Area": tf.constant([1], dtype=tf.int32),
        "RegionCategory": tf.constant([1], dtype=tf.int32),
        "TenureYears": tf.constant([0.5], dtype=tf.float32)
    }
    
    _ = index(dummy_input)

    index_save_path = os.path.join(version_dir, "bruteforce_index")
    tf.saved_model.save(index, index_save_path)
    logger.info(f"BruteForce model saved to {index_save_path}")

if __name__ == "__main__":
    train()
