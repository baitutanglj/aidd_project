import os
import pickle
import argparse
from time import time
from rdkit import Chem
from functools import partial
from multiprocessing import Pool, cpu_count
from utils import read_input_file, format_args


def sub_search(target_mol, sub_mol):
    match = target_mol.HasSubstructMatch(sub_mol)
    return match

def main(args):
    ref_mol = read_input_file(args.input_file)

    if ref_mol:
        with open(args.filename, 'rb') as f:
            database_df = pickle.load(f)
        database_df.rename(columns={'smiles': 'smiles', 'name_id': 'name', 'fingerprint': 'fp', 'mols': 'mol'}, inplace=True)

        func = partial(sub_search, sub_mol=ref_mol)
        # similarity
        cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK')) if os.getenv('SLURM_CPUS_PER_TASK') else cpu_count()
        print('cpus_per_task', cpus_per_task)
        print('cpu_count()', cpu_count())
        with Pool(cpus_per_task) as pool:
            database_df['match'] = pool.map(func, database_df['mol'])
        # filter
        database_df = database_df.loc[database_df['match']==True]
        database_df = database_df[:args.topk]
        database_df.drop(['fp', 'mol', 'match'], axis=1, inplace=True)

        database_df.to_csv(os.path.join(args.output_dir, f"out_sub_match_{args.outfilename}_{args.job_id}.csv"),
                           index=False, float_format='%.4f')
        print(database_df.head())

        return database_df

    else:
        return None


if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser(description="Calculate 2d morgan fingerprint similarity.")
    parser.add_argument("--config_file", type=str, help="Path to the JSON file containing parameters.",
                        default="/mnt/home/linjie/projects/AIDD/aidd_project/configs/linjie/sub_match/0.json")
    parser.add_argument("-j", "--job_id", type=str, help="slurm job id", default='123456')
    parser.add_argument("-o", "--output_dir", type=str, help="Output dir of result", default='./outputs')
    parser.add_argument("-fn", "--filename", type=str, help=".pkl database name", default='./data/sub_match/0.pkl')

    args = parser.parse_args()
    args = format_args(args)
    main(args)
    print(f"python finished run time:{round(time() - start_time, 5)} seconds")
