#!/bin/bash

#SBATCH -J sweep_latte2go                  # Job name
#SBATCH -o %x.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested
#SBATCH -n 1                             # Total number of mpi tasks requested
#SBATCH --cpus-per-task 64              # Number of CPUs each task
#SBATCH --mem 128000                    # Request memory allocation in MB
#SBATCH -t 12:00:00                      # Run time (hh:mm:ss)

cd ~/Projects/MultiOmicsGraphEmbedding
#ln -s ../Multiplex-Graph-Embedding/models models
#ln -s ../Multiplex-Graph-Embedding/data data

#wandb artifact cache cleanup 3000000

WANDB_AGENT_MAX_INITIAL_FAILURES=100 WANDB_AGENT_DISABLE_FLAPPING=true wandb agent jonnytran/LATTE2GO/d20ytrg1 & \
sleep 120 &&  wandb agent jonnytran/LATTE2GO/d20ytrg1 & \
sleep 120 &&  wandb agent jonnytran/LATTE2GO/d20ytrg1 && echo "done"
exit