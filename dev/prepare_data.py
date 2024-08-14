import json
import os
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

from llama_index import SQLDatabase, VectorStoreIndex
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.query_pipeline import QueryPipeline as QP
from llama_index.schema import TextNode
from llama_index.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.program import LLMTextCompletionProgram

# Constants
TABLEINFO_DIR = 'table_info_directory'
DB_FILE = "transactions.db"
# https://www.kaggle.com/datasets/rajatsurana979/comprehensive-credit-card-transactions-dataset
CSV_FILE = "transactions.csv"
TABLE_INDEX_DIR = "table_index_dir"

# Ensure necessary directories exist
Path(TABLEINFO_DIR).mkdir(parents=True, exist_ok=True)

######### Helper Functions #########

class TableInfo(BaseModel):
    """Information regarding a structured table."""
    table_name: str = Field(..., description="Table name (must use underscores, no spaces).")
    table_summary: str = Field(..., description="Short, concise summary of the table.")

def sanitize_column_name(col_name: str) -> str:
    """Remove special characters and replace spaces with underscores."""
    return re.sub(r"\W+", "_", col_name)

def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine, metadata_obj):
    """Create a SQL table from a Pandas DataFrame."""
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    table = Table(table_name, metadata_obj, *columns)
    metadata_obj.create_all(engine)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()

def _get_tableinfo_with_index(idx: int) -> TableInfo:
    """Retrieve table information file by index."""
    results_gen = Path(TABLEINFO_DIR).glob(f"{idx}_*")
    results_list = list(results_gen)
    
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(f"More than one file matching index: {list(results_gen)}")

def index_all_tables(sql_database: SQLDatabase, table_index_dir: str = "table_index_dir") -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    qp = QP(verbose=False)
    service_context = ServiceContext.from_defaults(callback_manager=qp.callback_manager)
    engine = sql_database.engine
    
    for table_name in sql_database.get_usable_table_names():
        print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = [tuple(row) for row in result]

            nodes = [TextNode(text=str(t)) for t in row_tups[:5]]

            index = VectorStoreIndex(nodes, service_context=service_context)
            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            print('Index already exists for table:', table_name)

######### Main Function #########

def main():
    # Load data from CSV
    # https://www.kaggle.com/datasets/rajatsurana979/comprehensive-credit-card-transactions-dataset
    df = pd.read_csv('https://raw.githubusercontent.com/divakaivan/finance_voice_assistant/main/dev/transactions.csv')
    # preprocessing
    df.drop(['Customer ID', 'Birthdate'], inplace=True, axis=1)
    df['Name'] = 'Ivan'
    df['Surname'] = 'Ivanov'
    df['Gender'] = 'M'

    # Initialize database and metadata
    engine = create_engine(f"sqlite:///{DB_FILE}")
    metadata_obj = MetaData()

    prompt_str = """\
    Give me a summary of the table with the following JSON format.

    - The table name must be unique to the table and describe it while being concise.
    - Do NOT output a generic table name (e.g., table, my_table).

    Table:
    {table_str}

    Summary: """

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        llm=OpenAI(model="gpt-3.5-turbo"),
        prompt_template_str=prompt_str,
    )

    df_str = df.head(10).to_csv()
    table_info = program(table_str=df_str)
    table_name = table_info.table_name
    print(f"Processed table: {table_name}")
    
    idx = 0
    out_file = f"{TABLEINFO_DIR}/{idx}_{table_name}.json"
    json.dump(table_info.dict(), open(out_file, "w"))

    # Load table info and create table
    tableinfo = _get_tableinfo_with_index(idx)
    print(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)

    # Create SQL database object
    sql_database = SQLDatabase(engine)

    # Index all tables
    index_all_tables(sql_database)

if __name__ == "__main__":
    main()
