import os
import sys
import yaml
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score, log_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ranking.common.dataset import load_ranking_data
from src.common.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def train_xgboost():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    model_dir = os.path.join("models", "ranking", "xgb_" + params['serving']['model_version'])
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Loading and preparing data for XGBoost...")
    train_df, test_df = load_ranking_data(
        params['data']['train_interactions_path'],
        params['data']['test_interactions_path'],
        params['data']['users_path'],
        params['data']['items_path'],
        num_negatives_per_positive=4,
        random_seed=params['random_seed']
    )

    features_to_exclude = ['CustomerCode', 'ProductCode', 'label', 'ProductName']
    feature_cols = [col for col in train_df.columns if col not in features_to_exclude]
    target_col = 'label'

    logger.info(f"Training features: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=params['random_seed'], stratify=y_train
    )

    dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
    dval = xgb.DMatrix(X_val_split, label=y_val_split)
    xgb_params = {
        'objective': 'binary:logistic', 
        'eval_metric': ['logloss', 'auc', 'ndcg'],
        'max_depth': 6,                 
        'eta': 0.1,                     
        'subsample': 0.8,               
        'colsample_bytree': 0.8,       
        'gamma': 0.1,
        'seed': params['random_seed'],
    }
    
    num_boost_round = 200 
    logger.info("Starting XGBoost training...")
    evals_result = {} 
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=True
    )
    

    logger.info("Calculating final validation metrics...")
    y_pred_val = model.predict(dval)
    
    final_auc = roc_auc_score(y_val_split, y_pred_val)
    final_logloss = log_loss(y_val_split, y_pred_val)
    
    logger.info(f"Final Validation AUC: {final_auc:.4f}")
    logger.info(f"Final Validation LogLoss: {final_logloss:.4f}")

    model_path = os.path.join(model_dir, "xgboost_model.json")
    model.save_model(model_path)
    logger.info(f"XGBoost model saved to {model_path}")
    
    history_path = os.path.join(model_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(evals_result, f)
    logger.info(f"Training history saved to {history_path}")

    feature_names_path = os.path.join(model_dir, "feature_names.pkl")
    with open(feature_names_path, "wb") as f:
        pickle.dump(feature_cols, f)

if __name__ == "__main__":
    train_xgboost()
