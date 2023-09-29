#!/bin/bash

#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# Thread count:
#SBATCH --cpus-per-task=16
# memory in MB
#SBATCH --mem=250000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=<error_path>/torch_ssl_%04a_stdout.txt
#SBATCH --error=<out_path>/torch_ssl_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=ssl_torch
#SBATCH --mail-user=<email>
#SBATCH --mail-type=ALL
#SBATCH --chdir=/<exp_path>/Self-Pseudo-Labeling
#SBATCH --array=<array>
#################################################


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip


. /home/fagg/tf_setup.sh
conda activate torch
wandb login key

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
run.py


