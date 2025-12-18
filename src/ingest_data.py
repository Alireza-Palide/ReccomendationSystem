import os
import yaml
import sys
import pandas as pd
from sqlalchemy import text


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.common.database import DatabaseConnector # type: ignore
from src.common.logger import setup_logging, get_logger # type: ignore

setup_logging()
logger = get_logger(__name__) 

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def ingest():
    config = load_config()
    db_config = config['database']
    connector = DatabaseConnector(db_config)
    
    logger.info("Extracting Customers...")
    query_customers = """
    SELECT 
        CustomerCode,
        TownName,
        GroupHeaderName,
        Area,
        Cluster,
        RegionCategory,
        -- Convert Date to meaningful feature (Years Active)
        DATEDIFF(year, EntryDateEn, GETDATE()) as TenureYears 
    FROM CustomerDataForRec
    """
    df_users = connector.extract_data(query_customers)
    
    # MLOps: Enforce types to prevent drift
    df_users['CustomerCode'] = df_users['CustomerCode'].astype(str)
    df_users['Cluster'] = df_users['Cluster'].astype(str) # Treat cluster as categorical, not numeric
    
    df_users.to_parquet("data/raw/customers.parquet", index=False)
    logger.info(f"Saved {len(df_users)} customers to data/raw/customers.parquet")

    # ---------------------------------------------------------
    # 2. EXTRACT PRODUCTS (Item Features)
    # ---------------------------------------------------------
    logger.info("Extracting Products...")
    query_products = """
    SELECT 
        ProductCode,
        ProductName,
        Price,
        ProductGroupHeader,
        ProductGroupName,
        IsBestSeller
    FROM ProductDataForRec
    """
    df_items = connector.extract_data(query_products)
    
    df_items['ProductCode'] = df_items['ProductCode'].astype(str)
    df_items['IsBestSeller'] = df_items['IsBestSeller'].astype(int)
    
    df_items.to_parquet("data/raw/products.parquet", index=False)
    logger.info(f"Saved {len(df_items)} products to data/raw/products.parquet")

    # ---------------------------------------------------------
    # 3. EXTRACT INTERACTIONS (The Positive Training Pairs)
    # ---------------------------------------------------------
    logger.info("Extracting Interactions...")

    # We join with Product/Customer tables just to be safe that 
    # we don't pull transactions for deleted users/items.
    query_interactions = """
    SELECT 
        cp.CustomerCode,
        cp.ProductCode,
        cp.Recency,
        cp.Frequency,
        cp.Amount
    FROM CustomerPurchaseForRec cp
    INNER JOIN CustomerDataForRec c ON cp.CustomerCode = c.CustomerCode
    INNER JOIN ProductDataForRec p ON cp.ProductCode = p.ProductCode
    """
    df_interactions = connector.extract_data(query_interactions)
    
    df_interactions['CustomerCode'] = df_interactions['CustomerCode'].astype(str)
    df_interactions['ProductCode'] = df_interactions['ProductCode'].astype(str)
    
    # Implicit Feedback Logic:
    # We will likely use 'Frequency' or 'Amount' as a weight in the loss function later.
    
    df_interactions.to_parquet("data/raw/interactions.parquet", index=False)
    logger.info(f"Saved {len(df_interactions)} interactions to data/raw/interactions.parquet")

if __name__ == "__main__":
    ingest()
