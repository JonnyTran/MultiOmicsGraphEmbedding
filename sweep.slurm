#!/bin/bash

#SBATCH -J sweep_latte2go                  # Job name
#SBATCH -o %x.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                             # Total number of mpi tasks requested
#SBATCH -t 05:10:00                      # Run time (hh:mm:ss)

cd ~/Projects/MultiOmicsGraphEmbedding
ln -s ../Multiplex-Graph-Embedding/models models
ln -s ../Multiplex-Graph-Embedding/data data

wandb agent jonnytran/latte-link-pred/7pv7akst
exit