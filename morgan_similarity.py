import os
import json
import pickle
import argparse
from time import time
from rdkit import Chem
from pathlib import Path
from functools import partial
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool, cpu_count
from utils import read_input_file, format_args

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def calculate_similarity(target_fp, ref_fp):
    """
    计算两个指纹之间的Tanimoto相似性
    """
    return DataStructs.TanimotoSimilarity(ref_fp, target_fp)


def filter_similar_molecules(ref_fp, database_df, topk):
    func = partial(calculate_similarity, ref_fp=ref_fp)
    # similarity
    # cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK')) if os.getenv('SLURM_CPUS_PER_TASK') else cpu_count()
    cpus_per_task = 10
    print('cpus_per_task', cpus_per_task)
    pool_time = time()
    with Pool(cpus_per_task) as pool:
        database_df['similarity'] = pool.map(func, database_df['fp'])
    print(f"pool file {round(time() - pool_time, 5)} seconds")
    # filter
    database_df.sort_values(by=['similarity'], ascending=False, inplace=True, ignore_index=True)
    database_df.drop(['fp', 'mol'], axis=1, inplace=True)
    database_df = database_df[:topk]
    return database_df


def main(args):
    """
    :param args:
    :return: database_df, columns: ['smiles', 'name', 'similarity']
    """
    ref_mol = read_input_file(args.input_file)
    if ref_mol:
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
        read_file_time = time()
        with open(args.filename, 'rb') as f:
            database_df = pickle.load(f)
        print(f"read file {round(time() - read_file_time, 5)} seconds")
        database_df.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'}, inplace=True)
        database_df = filter_similar_molecules(ref_fp, database_df, args.topk)
        database_df.to_csv(os.path.join(args.output_dir, f"out_morgan_similarity_{args.outfilename}_{args.job_id}.csv"),
                           index=False, float_format='%.4f')
        print(f"Found top {len(database_df)} molecules with similarity.")
        print(database_df.head())
        return database_df

    else:
        return None



if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser(description="Calculate 2d morgan fingerprint similarity.")
    parser.add_argument("--config_file", type=str, help="Path to the JSON file containing parameters.",
                        default="./configs/linjie/morgan_similarity/0.json")
    parser.add_argument("-j", "--job_id", type=str, help="slurm job id", default='123456')
    parser.add_argument("-o", "--output_dir", type=str, help="Output dir of result", default='./outputs')
    parser.add_argument("-fn", "--filename", type=str, help=".pkl database name", default='./data/morgan_similarity/1.pkl')
    # parser.add_argument("-i", "--input_file", type=str, help="Input file of molecule to be queried")
    # parser.add_argument("-d", "--database_dir", type=str, help=".pkl database directory of molecule")
    # parser.add_argument("--topk", type=int, help="top-K molecules to be save", default=1000)
    # parser.add_argument("-c", "--cpu_num", type=int, help="Number of CPU cores", default=28)

    args = parser.parse_args()
    args = format_args(args)
    print(args)
    main(args)
    print(f"python finished run time:{round(time()-start_time,5)} seconds")#28cpu:25.86545, 10cpu:26.06161
