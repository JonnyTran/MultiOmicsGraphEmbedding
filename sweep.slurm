#!/bin/bash

#SBATCH -J sweep_latte2go                  # Job name
#SBATCH -o %x.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                             # Total number of mpi tasks requested
#SBATCH --cpus-per-task 64              # Number of CPUs each task
#SBATCH --mem 128000                    # Request memory allocation in MB
#SBATCH -t 05:10:00                      # Run time (hh:mm:ss)

cd ~/Projects/MultiOmicsGraphEmbedding
ln -s ../Multiplex-Graph-Embedding/models models
ln -s ../Multiplex-Graph-Embedding/data data

wandb agent jonnytran/latte-link-pred/9mxnr1ag
exit