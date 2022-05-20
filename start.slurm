#!/bin/bash

#SBATCH -J latte2go                  # Job name
#SBATCH -o %x.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                             # Total number of mpi tasks requested
#SBATCH -t 05:10:00                      # Run time (hh:mm:ss)

cd ~/Projects/Multiplex-Graph-Embedding
#python ../MultiOmicsGraphEmbedding/run/run_mlm.py -y ../MultiOmicsGraphEmbedding/run/configs/bert_mlm_Protein.yaml --hours 5 &
#python ../MultiOmicsGraphEmbedding/run/run_mlm.py -y ../MultiOmicsGraphEmbedding/run/configs/bert_mlm_GO_term.yaml --gpu 1 --hours 5 && echo "done"

python ../MultiOmicsGraphEmbedding/run/latte_link.py -y ../MultiOmicsGraphEmbedding/run/configs/latte2go.yaml --num_gpus 1 --hours 5
exit