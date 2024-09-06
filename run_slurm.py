import os
import re
import json
import pickle
import time
import math
import argparse
import tempfile
import subprocess
import pandas as pd
from glob import glob
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from utils import execute_command, load_parameters, wait_until_file_generation
from merge_result import merge_output
import warnings
warnings.filterwarnings('ignore')

###  #SBATCH --cpus-per-task={args.cpus_per_task}

def write_slurm_script(args, filename):
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    python_script_dir, python_script = os.path.split(args.python_script)
    outfilename = Path(filename).stem
    basepath = os.path.split(os.path.realpath(__file__))[0]
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={args.job_name}
#SBATCH --nodes=1
#SBATCH --output={args.output_dir}/out_{outfilename}_%j.log
#SBATCH --error={args.output_dir}/error_{outfilename}_%j.log
\n
cd {basepath}
\n
source /shared/Programs/anaconda3/bin/activate {args.conda_env}
\n
# 记录开始时间
start_time=$(date +%s.%N)
\n
srun python {python_script} --config_file {args.config_file} --j $SLURM_JOB_ID -o {args.output_dir} -fn {filename}
\n
# 记录结束时间
end_time=$(date +%s.%N)
# 计算运行时长
runtime=$(echo "$end_time - $start_time" | bc)
echo "Job $SLURM_JOB_ID runtime: $runtime seconds"
"""

    shell_name = f"run_job_{int(time.time())}.sh"
    with open(f"./slurm_config/{shell_name}", 'w') as f:
        f.write(slurm_script_content)
    return {'shell_name': shell_name,
            'output': f"{args.output_dir}/out_{outfilename}",
            'error': f"{args.output_dir}/error_{outfilename}"}


def submit_slurm_job(shell_name):
    result = subprocess.run(['sbatch', f"./slurm_config/{shell_name}"], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully with Job ID: {job_id}")
        os.remove(f"./slurm_config/{shell_name}")
        return job_id
    else:
        print("Failed to submit job.")
        print(result.stderr)
        return None

def check_job_status(job_ids):
    cmd = ['squeue', '--job', ','.join(job_ids)]#显示当前排队和正在运行的作业
    result = subprocess.run(cmd, capture_output=True, text=True)
    result = result.stdout.strip().split('\n')
    header = re.split(r'\s{1,}', result[0].strip())
    data = [re.split(r'\s{1,}', line.strip()) for line in result[1:] if line]
    df = pd.DataFrame(data, columns=header)
    return df

def check_for_errors(job_infos):
    for job in job_infos:
        if os.path.exists(job['error']):
            with open(f"{job['error']}_{job['job_id']}.log", 'r') as file:
                error_content = file.read().strip()
                if error_content:
                    return job['id'], error_content
    return None, None

def get_job_output(job_id, base_output_log):
    log_file = f"{base_output_log}_{job_id}.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            output = f.read()
        return output
    else:
        print(f"Log file {log_file} does not exist.")
        return None


def main(args):
    filenames = glob(args.database_dir+'/*')
    job_infos = []
    for filename in filenames:
        job_info = write_slurm_script(args, filename)
        job_id = submit_slurm_job(job_info["shell_name"])
        job_info['job_id'] = job_id
        job_infos.append(job_info)
    print([job['job_id'] for job in job_infos])

    while job_infos:
        print("check_job_status")
        status_df = check_job_status([job['job_id'] for job in job_infos])
        if len(status_df)==0:
            print("All jobs have completed.")
            break
        else:
            print('status_output', status_df)
            job_id, error_content = check_for_errors(job_infos)
            if error_content:
                print(f"Job {job_id} encountered an error:\n{error_content}")
                exit(1)
            print("Some jobs are still running or in queue...")
            time.sleep(30)  # 等待30秒再检查


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run slurm task")
    parser.add_argument("-j", "--job_name", type=str, help="slurm task job name", default="AIDD_task")
    parser.add_argument("-c", "--cpus_per_task", type=int, help="--cpus-per-task", default=2)
    parser.add_argument("-o", "--output_dir", type=str, help="slurm output log dir", default="./outputs")
    parser.add_argument("-p", "--python_script", type=str, help="python script to srun", default="./morgan_similarity.py")
    parser.add_argument("-f", "--config_file", type=str, help="Path to the JSON file containing  srun .py parameters.",
                        default="./configs/linjie/morgan_similarity/0.json")
    parser.add_argument("-d", "--database_dir", type=str, help="database directory of molecules files",
                        default="./data/morgan_similarity")
    parser.add_argument("--conda_env", type=str, help="conda env name", default="/shared/Programs/anaconda3/envs/AIDD")
    args = parser.parse_args()
    now = datetime.now()
    formatted_time = 'tmp'+now.strftime("%Y%m%d%H%M%S")+'_'
    args.output_dir = tempfile.mkdtemp(suffix=None, prefix=formatted_time, dir=args.output_dir)
    print(f"Output directory: {args.output_dir}")
    args.topk = load_parameters(args.config_file)['topk']

    main(args)
    merge_output(args.output_dir, args.topk, args.python_script)




