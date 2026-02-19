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
        self.config = config
        self.engine = self._create_engine()

    def _create_engine(self):
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.config.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
            f"SERVER={self.config['server']};"
            f"DATABASE={self.config['database']};"
            f"Trusted_Connection={'yes' if self.config.get('trusted', True) else 'no'};"
            f"UID={self.config.get('sa')};"
            f"PWD={self.config.get('55741437')};"
        )
        
        return sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

    def extract_data(self, query: str, chunksize: int = 100000) -> pd.DataFrame:
    
        logger.info(f"Executing query: {query[:50]}...")
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Extraction complete. Loaded {len(df)} rows.")
            return df
        except Exception as e:
            logger.error("Database Error: {e}", exc_info=True)
            raise e
