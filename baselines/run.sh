python run_her.py --num_timesteps 15000  --n_arms 5 
python run_her.py --num_timesteps 15000  --n_arms 5 --relative_goals True 
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type sums
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type fc
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type layered --hidden 20
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type sums --relative_goals True 
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type fc --relative_goals  True  
python run_her.py --num_timesteps 15000 --parts True --n_arms 5 --conn_type layered --relative_goals True --hidden 20
