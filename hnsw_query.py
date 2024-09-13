import os
import gc
import hnswlib
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from time import time
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from utils import read_input_file
from hnsw_utils import HNSW_Query
from postgre_utils import Zinc_Database


# ---------加载已创建的hnsw index 并进行查询--------#
ef_construction = 200
M = 16
space = 'cosine'
index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1.bin"
filepaths = glob("/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/*.pkl")
myhnsw = HNSW_Query(filepaths, ef_construction=ef_construction, M=M, space=space, hnsw_index_path=index_path)
ef = 100
num_threads = cpu_count()
labels, distances = myhnsw.query_hnsw_index(query='Clc1c(C(C)(C)C)cccc1', topk=1000, ef=100, num_threads=num_threads)

# ---------连接 PostgreSQL数据库 并进行查询--------#
db_name = "testdb" # 数据库名称
table_name = "similarity2d"
db_username = 'linjie'
db_password = 'lj123..'
db_host = '192.168.109.38'  # 数据库服务器地址
db_port = '5432'  # PostgreSQL 默认端口
zinc_database = Zinc_Database(db_name, table_name, db_username, db_password, db_host, db_port)
result_df = zinc_database.query(query_idx_list=list(labels), score_list=distances)
savepath = "./outputs/hnsw_similarity2d.csv"
result_df.to_csv(savepath, index=False)

