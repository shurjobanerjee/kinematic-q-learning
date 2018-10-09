# 
python run_her.py --num_timesteps 15  --hidden 16 --n_arms 2  --env Arm
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 2  --env Arm --normalized True
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 2 --parts diff --env Arm
#
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 6  --env Arm
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 6  --env Arm --normalized True
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 6 --parts diff --env Arm
#
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 10  --env Arm
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 10  --env Arm --normalized True
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 10 --parts diff --env Arm
#
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 15  --env Arm
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 15  --env Arm --normalized True
#python run_her.py --num_timesteps 150000  --hidden 16 --n_arms 15 --parts diff --env Arm
#
#python run_her.py --num_timesteps 150000  --hidden 256 --env Hand
#python run_her.py --num_timesteps 150000  --hidden 256 --env Hand --normalized True
#python run_her.py --num_timesteps 150000  --hidden 212 --parts diff --env Hand
#
#python run_her.py --num_timesteps 150000  --hidden 256 --env Fetch
#python run_her.py --num_timesteps 150000  --hidden 256 --env Fetch --normalized True
#python run_her.py --num_timesteps 150000  --hidden 212 --parts diff --env Fetch
#
# 
## Diagnoes the problem (FETCH 2D)
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints "" 
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints "" 
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints True
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints True
#
#python run_her.py --normalized True --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints "" 
#python run_her.py --normalized True --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints "" 
#python run_her.py --normalized True --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints True
#python run_her.py --normalized True --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints True
#
#
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints ""    --parts diff  
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints ""    --parts diff 
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions ""   --constraints True  --parts diff 
#python run_her.py --num_timesteps 150000 --hidden 16 --env Fetch --collisions True --constraints True  --parts diff 
#
