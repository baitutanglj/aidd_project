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

# ------------创建hnsw index-----------#
ef_construction = 200
M = 16
space = 'cosine'
filepaths = glob("/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/*.pkl")
myhnsw = HNSW_Query(filepaths, ef_construction=ef_construction, M=M, space=space)
for filepath in filepaths:
    print(filepath)
    myhnsw.create_hnsw_index(filepath)
    gc.collect()

index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1.bin"
myhnsw.save_hnsw_index(index_path)
