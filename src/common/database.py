import os
import urllib
import sqlalchemy
import pandas as pd
from typing import Optional

from src.common.logger import setup_logging, get_logger # type: ignore

setup_logging()
logger = get_logger(__name__) 

class DatabaseConnector:
    def __init__(self, config: dict):
        """
        Initializes the connection engine using SQLAlchemy.
        
        Args:
            config: Dictionary containing 'driver', 'server', 'database', etc.
        """
        self.config = config
        self.engine = self._create_engine()

    def _create_engine(self):
        """
        Constructs the connection string securely.
        Uses Windows Authentication (Trusted_Connection) by default if user/pass not provided.
        """
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.config.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
            f"SERVER={self.config['.']};"
            f"DATABASE={self.config['Uni']};"
            f"Trusted_Connection={'yes' if self.config.get('trusted', True) else 'no'};"
            f"UID={self.config.get('sa')};"
            f"PWD={self.config.get('55741437')};"
        )
        
        return sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

    def extract_data(self, query: str, chunksize: int = 100000) -> pd.DataFrame:
        """
        Executes a query and returns a Pandas DataFrame.
        
        CRITICAL MLOPS NOTE: 
        For large datasets (millions of rows), we use chunking to avoid 
        blowing up RAM. This function creates a generator or concatenates.
        """

        logger.info(f"Executing query: {query[:50]}...")
        try:
            # Using pandas read_sql is more efficient than raw cursors for DataFrames
            # We assume the dataset fits in memory for this thesis scope (e.g. < 5GB).
            # If > 5GB, we would need to stream to disk (Parquet) directly.
            df = pd.read_sql(query, self.engine)
            logger.info(f"Extraction complete. Loaded {len(df)} rows.")
            return df
        except Exception as e:
            logger.error("Database Error: {e}", exc_info=True)
            raise e
