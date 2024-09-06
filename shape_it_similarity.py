import os
import argparse
import time
from pathlib import Path
import pandas as pd
from utils import format_args, execute_command, wait_until_file_generation


def main(args):
    name = Path(args.filename).stem
    output_file = os.path.join(args.output_dir,  f"out_3D_similarity_{name}_{args.job_id}.sdf")
    score_file = os.path.join(args.output_dir, f"out_3D_similarity_{name}_{args.job_id}.csv")
    arguments = ['-r', args.reference, '-d', args.filename,
                 '-o', output_file, '-s', score_file, '--best', args.topk, '--rankBy', 'TANIMOTO']
    result = execute_command('/shared/Programs/shape-it/bin/shape-it', arguments, check=False, location=None)
    # result = execute_command('./shape-it/bin/shape-it', arguments, check=False, location=None)
    result_flag = wait_until_file_generation(score_file)
    if result_flag:
        print("shape-it finished successfully")
        return output_file, score_file
    else:
        return None, None





if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser(description="vina docking")
    parser.add_argument("--config_file", type=str, help="Path to the JSON file containing parameters.",
                        default="/mnt/home/linjie/projects/AIDD/aidd_project/configs/linjie/shape_it/0.json")
    parser.add_argument("-j", "--job_id", type=str, help="slurm job id", default='123456')
    parser.add_argument("-o", "--output_dir", type=str, help="Output dir of result", default='./outputs')
    parser.add_argument("-fn", "--filename", type=str, help=".pkl database name", default='./data/shape_it/0.sdf')
    args = parser.parse_args()
    args = format_args(args)
    main(args)
    print(f"python finished run time:{round(time() - start_time, 5)} seconds")
