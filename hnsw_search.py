import os
import hnswlib
import pickle
import numpy as np
import pandas as pd
from time import time
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((1,), dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

def get_data(filepaths):
    df, fp = pd.DataFrame(), []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            df_tmp = pickle.load(f)
            df_tmp.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'},
                               inplace=True)
        fp_tmp = np.vstack(df_tmp['fp'])
        df_tmp.drop(columns=['fp', 'mol'], inplace=True)
        df = df._append(df_tmp)
        fp.append(fp_tmp)
    fp = np.vstack(fp)
    df.reset_index(inplace=True)
    return df, fp


def create_hnsw_index(fp, ef_construction=200, M=16, space='cosine'):
    # Creating HNSW index
    # Declaring index
    num_elements, dim = fp.shape  # number of vectors to index, dimensionality of the vectors
    p = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    start_time = time()
    p.add_items(fp, ids=range(num_elements), num_threads=-1)
    finish_time = round(time()-start_time, 5)
    print(f"p.add_items finish_time: {finish_time}")
    return p

def load_hnsw_index(num_elements, dim, index_path, space='cosine'):
    # Reiniting, loading the index
    p = hnswlib.Index(space=space, dim=dim)  # the space can be changed - keeps the data, alters the distance function.
    print("\nLoading index from 'first_half.bin'\n")
    # Increase the total capacity (max_elements), so that it will handle the new data
    p.load_index(index_path, max_elements=num_elements)
    return p

def query_hnsw_index(query_fp, k):
    start_time = time()
    labels, distances = p.knn_query(query_fp, k=k)
    finish_time = round(time() - start_time, 5)
    print(f"p.knn_query: {finish_time}")
    return labels, distances


filepaths = ['/mnt/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/1.pkl']
df, fp = get_data(filepaths)#26.3G-8G=10.3G   #512bits  17.0-6.88=10.12G
p = create_hnsw_index(fp)# <1G   #512bits  17.1-17.0=0.1
num_elements, dim = fp.shape
index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1.bin"
p.save_index(index_path)
del p#29.5-33.4=-3.9   #512bits
# del fp#21.9-29.5=-7.6
# del df #25.8-21.9=3.9
p = load_hnsw_index(num_elements, dim, index_path, space='cosine')#25.8-21.9=3.9
ef = 100
p.set_ef(ef)
p.set_num_threads(cpu_count())
query_fp = get_fingerprint('Clc1c(C(C)(C)C)cccc1')
labels, distances = query_hnsw_index(query_fp, 1000)#0.03726   #512bits  0.01125
labels, distances = query_hnsw_index(query_fp, 10000)#0.10952   #512bits  0.05876
labels, distances = query_hnsw_index(query_fp, 100000)#0.47983   #512bits  0.26503
labels, distances = query_hnsw_index(query_fp, 200000)#0.70328   #512bits  0.42814

labels, distances = query_hnsw_index(fp[:100], 1000)#0.57019   #512bits  0.1944
labels, distances = query_hnsw_index(fp[:100], 10000)#6.18373   #512bits  3.14727
labels, distances = query_hnsw_index(fp[:100], 100000)#42.46232   #512bits  24.40879
labels, distances = query_hnsw_index(fp[:100], 200000)#65.39292   #512bits  38.86758





# filepaths = ['/mnt/home/linjie/projects/AIDD/aidd_project/data/morgan_similarity/1_512bit.pkl']
# df, fp = get_data(filepaths)#29.5G-7.77G=21.8G   #512bits  17.0-6.88=10.12G
# p = create_hnsw_index(fp)#33.4-29.5=3.9G   #512bits  17.1-17.0=0.1
# num_elements, dim = fp.shape
# index_path = "/mnt/home/linjie/projects/AIDD/aidd_project/data/hnsw_index/1_512bits.bin"
# p.save_index(index_path)
# del p#29.5-33.4=-3.9   #512bits
# # del fp#21.9-29.5=-7.6
# # del df #25.8-21.9=3.9