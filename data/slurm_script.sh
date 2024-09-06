#!/bin/bash

JOB_NAME=$1
CPUS_PER_TASK=$2
OUTPUT_LOG="$3_$(date +%Y%m%d%H%M%S)_${SLURM_JOB_ID}.log"
CONDA_ENV=$4

#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # 每个节点运行一个任务
#SBATCH --cpus-per-task=${CPUS_PER_TASK}    # 每个任务使用?个CPU核心
#SBATCH --output=${OUTPUT_LOG} # 标准输出和错误日志文件


source activate ${CONDA_ENV}
srun python morgan_similarity.py config.json
