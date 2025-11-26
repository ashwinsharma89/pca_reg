"""
Database Connector for PCA-Agent
Supports multiple database backends for data ingestion
"""

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Handles database connections and data retrieval
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database connector
        
        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - PostgreSQL: postgresql://user:password@host:port/database
                - MySQL: mysql+pymysql://user:password@host:port/database
                - SQL Server: mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
                - BigQuery: bigquery://project_id/dataset_id
        """
        self.connection_string = connection_string
        self.engine = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DataFrame with query results
        """
        if not self.engine:
            self.connect()
        
        try:
            if params:
                df = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                df = pd.read_sql_query(query, self.engine)
            
            logger.info(f"Query executed successfully. Rows returned: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def load_campaign_data(
        self,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load campaign data from database
        
        Args:
            table_name: Name of the table to query
            query: Custom SQL query (overrides table_name)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with campaign data
        """
        if query:
            # Use custom query
            sql_query = query
        elif table_name:
            # Build query from table name
            sql_query = f"SELECT * FROM {table_name}"
            
            # Add date filters if provided
            if start_date and end_date:
                sql_query += f" WHERE date >= '{start_date}' AND date <= '{end_date}'"
            elif start_date:
                sql_query += f" WHERE date >= '{start_date}'"
            elif end_date:
                sql_query += f" WHERE date <= '{end_date}'"
        else:
            raise ValueError("Either table_name or query must be provided")
        
        logger.info(f"Loading campaign data with query: {sql_query}")
        return self.execute_query(sql_query)
    
    def save_results(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'replace'
    ):
        """
        Save analysis results back to database
        
        Args:
            df: DataFrame to save
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        if not self.engine:
            self.connect()
        
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Results saved to table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise


# Example usage functions
def load_from_bigquery(
    project_id: str,
    dataset_id: str,
    table_name: str,
    credentials_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from Google BigQuery
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Table name
        credentials_path: Path to service account JSON
        
    Returns:
        DataFrame with campaign data
    """
    from google.cloud import bigquery
    from google.oauth2 import service_account
    
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = bigquery.Client(credentials=credentials, project=project_id)
    else:
        client = bigquery.Client(project=project_id)
    
    query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_name}`
    """
    
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df)} rows from BigQuery")
    
    return df


def load_from_snowflake(
    account: str,
    user: str,
    password: str,
    database: str,
    schema: str,
    warehouse: str,
    table_name: str
) -> pd.DataFrame:
    """
    Load data from Snowflake
    
    Args:
        account: Snowflake account identifier
        user: Username
        password: Password
        database: Database name
        schema: Schema name
        warehouse: Warehouse name
        table_name: Table name
        
    Returns:
        DataFrame with campaign data
    """
    import snowflake.connector
    
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    
    conn.close()
    logger.info(f"Loaded {len(df)} rows from Snowflake")
    
    return df


# Example configuration templates
DATABASE_CONFIGS = {
    'postgresql': {
        'connection_string': 'postgresql://user:password@localhost:5432/campaign_db',
        'example_query': 'SELECT * FROM campaigns WHERE date >= CURRENT_DATE - INTERVAL \'30 days\''
    },
    'mysql': {
        'connection_string': 'mysql+pymysql://user:password@localhost:3306/campaign_db',
        'example_query': 'SELECT * FROM campaigns WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)'
    },
    'bigquery': {
        'connection_string': 'bigquery://project-id/dataset-id',
        'example_query': 'SELECT * FROM campaigns WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)'
    },
    'snowflake': {
        'connection_string': 'snowflake://user:password@account/database/schema?warehouse=warehouse_name',
        'example_query': 'SELECT * FROM campaigns WHERE date >= DATEADD(day, -30, CURRENT_DATE())'
    }
}
