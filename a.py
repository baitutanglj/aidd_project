import os
import json
import pickle
import argparse
from rdkit import Chem
from functools import partial
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool, cpu_count
from utils import read_input_file, load_parameters


def main(args):
    cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    print('cpus_per_task', cpus_per_task, type(cpus_per_task))
    print('cpu_count()', cpu_count())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate 2d morgan fingerprint similarity.")
    parser.add_argument("--config_file", type=str, help="Path to the JSON file containing parameters.",
                        default="/mnt/home/linjie/projects/AIDD/aidd_project/configs/linjie/morgan_similarity/0.json")
    parser.add_argument("-j", "--job_id", type=str, help="slurm job id")

    args = parser.parse_args()
    job_id = args.job_id
    print('args.job_id:', args.job_id)
    args = argparse.Namespace(**load_parameters(args.config_file))
    args.job_id = job_id

    main(args)
