#!/bin/bash

#SBATCH -J latte2go                  # Job name
#SBATCH -o %x.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                             # Total number of mpi tasks requested
#SBATCH -t 05:10:00                      # Run time (hh:mm:ss)

cd ~/Projects/MultiOmicsGraphEmbedding
ln -s ../Multiplex-Graph-Embedding/models models

wandb agent jonnytran/latte-link-pred/486nop6y
exit