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


class HNSW_Query:
    def __init__(self, filepaths_example, ef_construction=200, M=16, space='cosine', hnsw_index_path=None):
        num_elements, dim = self.get_dim(filepaths_example)
        self.num_elements = num_elements #number of vectors to index
        self.dim = dim # number of dimensionality of the vectors
        self.ef_construction = ef_construction
        self.M = M
        self.space = space
        self.data_count = 0
        # Creating HNSW index
        self.p = hnswlib.Index(space=self.space, dim=self.dim)  # possible options are l2, cosine or ip
        if hnsw_index_path:
            self.load_hnsw_index(hnsw_index_path)
        else:
            self.p.init_index(max_elements=self.num_elements, ef_construction=self.ef_construction, M=self.M)

    def get_dim(self, filepaths_example):
        df = pd.read_pickle(filepaths_example[0])
        df.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'},
                  inplace=True)
        dim = np.array(df.loc[0, 'fp']).shape[0]
        num_elements = len(df) * len(filepaths_example)
        del df
        return num_elements, dim

    def create_hnsw_index(self, filepath):
        fp, ids_list = self.get_data(filepath)
        self.data_count += len(fp)
        # Declaring index
        start_time = time()
        self.p.add_items(fp, ids=ids_list, num_threads=-1)
        finish_time = round(time() - start_time, 5)
        del fp
        gc.collect()
        print(f"p.add_items finish_time: {finish_time}")

    def get_fp_array(self, fp):
        fingerprint_array = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, fingerprint_array)
        return fingerprint_array

    def get_data(self, filepath):
        df = pd.read_pickle(filepath)
        df.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'},
                           inplace=True)
        pool = Pool(cpu_count())
        fp = pool.apply_async(self.get_fp_array, df['fp'])
        pool.close()
        pool.join()
        fp = np.vstack(fp)
        ids_list = [i+self.data_count for i in range(len(df))]
        del df
        gc.collect()
        print("get data finished")
        return fp, ids_list

    def save_hnsw_index(self, save_path):
        self.p.save_index(save_path)

    def load_hnsw_index(self, index_path):
        # Reiniting, loading the index
        print("\nLoading index from 'first_half.bin'\n")
        # Increase the total capacity (max_elements), so that it will handle the new data
        self.p.load_index(index_path, max_elements=self.num_elements)

    def get_fingerprint(self, query):
        mol = read_input_file(query)
        arr = None
        if mol:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((1,), dtype=np.int32)
            AllChem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr

    def query_hnsw_index(self, query, topk, ef, num_threads=-1):
        start_time = time()
        self.p.set_ef(ef)
        self.p.set_num_threads(num_threads)
        query_fp = self.get_fingerprint(query)
        labels, distances = self.p.knn_query(query_fp, k=topk)
        finish_time = round(time() - start_time, 5)
        print(f"p.knn_query: {finish_time}")
        return labels, distances



if __name__ == "__main__":
    #------------创建hnsw index-----------#
    ef_construction=200
    M=16
    space='cosine'
    filepaths = glob("/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/*.pkl")
    myhnsw = HNSW_Query(filepaths, ef_construction=ef_construction, M=M, space=space)
    for filepath in filepaths[:1]:
        print(filepath)
        myhnsw.create_hnsw_index(filepath)
        gc.collect()


    index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1.bin"
    myhnsw.save_hnsw_index(index_path)
    del myhnsw

    #---------加载已创建的hnsw index 并进行查询--------#

    # ef_construction = 200
    # M = 16
    # space = 'cosine'
    # index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1.bin"
    # filepaths = glob("/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/*.pkl")
    # myhnsw = HNSW_Query(filepaths, ef_construction=ef_construction, M=M, space=space, index_path)
    # ef = 100
    # num_threads = cpu_count()
    # labels, distances = myhnsw.query_hnsw_index(query='Clc1c(C(C)(C)C)cccc1', topk=1000, ef=100, num_threads=num_threads)



