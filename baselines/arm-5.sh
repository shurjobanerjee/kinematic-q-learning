#!/bin/bash -i

source ~/.bashrc
source ../setup.sh

# Baseline
#for5 python run_her.py --n_arms 5 --num_timesteps 15000 
#for5 python run_her.py --n_arms 5 --num_timesteps 15000 --normalized True

# Ours
#for5 python run_her.py --n_arms 5 --parts diff --num_timesteps 15000 
#for5 python run_her.py --n_arms 5 --parts diff --num_timesteps 15000 --normalized True



python run_her.py --n_arms 5 --seeds 6,7,8,9,10 --parts diff --num_timesteps 15000 
python run_her.py --n_arms 5 --seeds 6,7,8,9,10 --parts diff --num_timesteps 15000 --normalized True
