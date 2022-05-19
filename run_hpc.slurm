#!/bin/bash

#SBATCH -J bert_mlm_Protein_GO                  # Job name
#SBATCH -o %A.%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                   # Queue name
#SBATCH -N 1                             # Total number of nodes requested (56 cores/node)
#SBATCH -n 24                             # Total number of mpi tasks requested
#SBATCH -t 05:10:00                      # Run time (hh:mm:ss)

cd ~/Projects/Multiplex-Graph-Embedding
python ../MultiOmicsGraphEmbedding/run/run_mlm.py -y ../MultiOmicsGraphEmbedding/run/configs/bert_mlm_Protein.yaml --hours 5 &
python ../MultiOmicsGraphEmbedding/run/run_mlm.py -y ../MultiOmicsGraphEmbedding/run/configs/bert_mlm_GO_term.yaml --gpu 1 --hours 5 && echo "done"
exit