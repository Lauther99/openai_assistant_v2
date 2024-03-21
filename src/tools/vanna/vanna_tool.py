import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.settings import Settings
from src.components.llms.llms import LLMs

import pyodbc
import pandas as pd
from vanna.remote import VannaDefault
from typing import List, Dict, Optional


class VannaTool:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        # Configuraciones basicas
        if settings is None:
            self.settings = Settings()  
        else:
            self.settings = settings

        # Configuraciones de la db
        driver = self.settings.sql.db_driver
        server = self.settings.sql.db_host
        db_name = self.settings.sql.db_name
        user = self.settings.sql.db_user
        pwd = self.settings.sql.db_pwd

        self.conn = pyodbc.connect(
            f"DRIVER={driver};SERVER={server};DATABASE={db_name};UID={user};PWD={pwd}"
        )

        # Configuraciones del modelo de Vanna
        self.vn = VannaDefault(
            model=self.settings.vanna.model, api_key=self.settings.vanna.api_key
        )
        self.vn.run_sql = self.run_sql
        self.vn.run_sql_is_set = True

    def run_sql(self, sql: str) -> pd.DataFrame:
        df = pd.read_sql_query(sql, self.conn)
        return df

    def get_training_data(self) -> pd.DataFrame:
        return self.vn.get_training_data()

    def generate_sql(self, question: str) -> str:
        q = f"{question}, use the names of columns given exclusively, do not hallucinate or create new ones."
        return self.vn.generate_sql(q)
    
    def generate_summary(self, question: str, dataframe: pd.DataFrame) -> str:
        return self.vn.generate_summary(question, dataframe)

    def feed_ddl(self, ddl: str):
        self.vn.train(ddl=ddl)

    def feed_examples(self, question: str, sql: str):
        self.vn.train(question=question, sql=sql)

    def feed_documentation(self, documentation: str):
        self.vn.train(ddl=documentation)

    def remove_all_data(self):
        training_data = self.get_training_data()
        ids = []
        for _, row in training_data.iterrows():
            ids.append(row["id"])
        for id in ids:
            self.vn.remove_training_data(id)

    def remove_single_data(self, id):
        self.vn.remove_training_data(id)
