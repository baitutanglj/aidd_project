import psycopg2
import numpy as np
import pandas as pd
from glob import glob
from sqlalchemy import create_engine


class Zinc_Database:
    def __init__(self, db_name, table_name, db_username, db_password, db_host, db_port):
        self.db_name = db_name
        self.table_name = table_name
        self.db_username = db_username
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.data_count = 0
        self.engine = self.connect_db_engine()

    def connect_db_engine(self):
        # 创建连接引擎
        engine = create_engine(f'postgresql+psycopg2://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}')
        return engine

    def dump_db(self, filepath):
        database_df = pd.read_pickle(filepath)
        database_df.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'},
                           inplace=True)
        print(database_df.head())
        # database_df['fp'] = database_df['fp'].apply(lambda x: np.array(x))
        # database_df['id'] = [self.data_count+i for i in range(len(database_df))]
        database_df.index = pd.RangeIndex(start=self.data_count, stop=self.data_count+len(database_df), step=1)
        database_df.drop(columns=['fp', 'mol'], inplace=True)
        # 将数据框写入数据库中的表，追加数据到已有表
        database_df.to_sql(self.table_name, self.engine, if_exists='append', index=True, index_label='index')
        self.data_count += len(database_df)
        print(f"dump database {filepath} ok")


    def query(self, query_idx_list, score_list):
        query = f"SELECT index, smiles, name FROM {self.table_name} WHERE index IN %s"
        result_df = pd.read_sql_query(query, self.engine, params=(tuple(query_idx_list),))
        result_df['score'] = score_list
        return result_df


if __name__=="__main__":
    db_name = "testdb" # 数据库名称
    table_name = "similarity2d"
    db_username = 'linjie'
    db_password = 'lj123..'
    db_host = '192.168.109.38'  # 数据库服务器地址
    db_port = '5432'  # PostgreSQL 默认端口
    zinc_database = Zinc_Database(db_name, table_name, db_username, db_password, db_host, db_port)
    filepaths = glob("/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/*.pkl")
    for filepath in filepaths:
        zinc_database.dump_db(filepath)

    query_idx_list = [1,23,3]
    score_list = [0.8, 0.5,0.3]
    result_df = zinc_database.query(query_idx_list, score_list)
    print(result_df)