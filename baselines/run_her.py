import click
import keyword2cmdline
import subprocess
import mujoco_py
from os.path import join

@keyword2cmdline.command
def main(num_timesteps=5000, play=False, log=True, parts='None', n_arms=2, env='2d', hidden=16, identifier='', **kwargs):
    
    gym_assets = "/z/home/shurjo/projects/kinematic-q-learning/gym/gym/envs/robotics/assets"
    if env == 'Arm':
        model = mujoco_py.load_model_from_path(join(gym_assets, "fetch/reach-actuated.xml"))
        n_arms = len(model.actuator_names)
    elif env == "Hand":
        model = mujoco_py.load_model_from_path(join(gym_assets,  "hand/reach.xml"))
        n_arms = len(model.actuator_names)

    # Governs whether to show a test simulation
    play = "" if not play else "--play"

    # Differentiate the method
    method = 'baseline' if parts is "None" else 'ours-{}'.format(parts)
    relative_goals = True #True if parts is "None" else False
    method = "{}-rel_goals-{}".format(method, relative_goals)

    # Experiment identifier if provided
    if identifier:
        method = '{}-{}'.format(method, identifier)

    # Folder to store logs in
    logs = "logs/{}/{}/{}/{}".format(env, n_arms, hidden, method)
    
    # Storing the logs requires the setting of an environmental variable
    store_logs = "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) \
                        if log else "OPENAI_LOG_FORMAT=stdout"

    if env == 'Fetch':
        command = """
                   {} 
                   mpirun -np 19
		   python -m baselines.run 
                   --alg=her 
                   --env=FetchReachAct-v1 
                   --num_timesteps={} 
                   --n_arms {}
	 	   --num_env=2 
                   {} 
                   """.format(store_logs, num_timesteps, n_arms, play)
    elif env == "Hand":
        command = """
                   {} 
                   mpirun -np 19
                   python -m baselines.run 
                   --alg=her 
                   --env=HandReach-v0 
                   --num_timesteps={} 
                   --n_arms {}
	           --num_env=2 
                   {} 
                   """.format(store_logs, num_timesteps, n_arms, play)
    else:
        command = """
                   {} 
                   python -m baselines.run 
                   --alg=her 
                   --env=Arm-v0 
                   --num_timesteps={} 
                   --n_arms {} 
                   {}
                   """.format(store_logs, num_timesteps, n_arms, play)

    # To a normal looking sentence
    command = " ".join(command.split())
        
    # Add kwargs 
    kwargs['hidden'] = hidden
    kwargs['parts']  = parts
    kwargs['relative_goals'] = relative_goals
    kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])

    # Kwargs
    command = "{} {}".format(command, kwargs_args)
    print(command)
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
