import click
import keyword2cmdline
import subprocess
import mujoco_py
import os
from os.path import join, abspath

@keyword2cmdline.command
def main(num_timesteps=5000, play=True, log=True, parts='None', n_arms=2, env='2d', hidden=16, identifier='', normalized=False, parallel=False, save_path=False,  constraints=False, collisions=False, load_path=False, **kwargs):
    
    # Governs whether to show a test simulation
    play = "" if not play else "--play"

    # Differentiate the method
    method = 'baseline' if parts is "None" else 'our'
    relative_goals = True #True if parts is "None" else False
    method = "{}-{}-rel_goals-{}-normalized-{}".format(env, method, relative_goals, normalized)

    # Model file name changes based on params
    model_xml_path = None
    gym_assets = "../gym/gym/envs/robotics/assets"
    n_arms_str = None
    if env == 'Fetch':
        #if not constraints and not collisions:
        #    xml_name = "fetch/reach-actuated.3arm-continous.xml" 
        #elif constraints and not collisions:
        #    xml_name = "fetch/reach-actuated.3arm-constrained.xml"
        #else:
        xml_name = "fetch/reach-actuated.xml"

        #xml_name = "fetch/reach-actuated.2d.collisions_{}_constraints_{}.xml".format(\
                             #collisions, constraints)
        model_xml_path = join(gym_assets, xml_name)
        model_xml_path = abspath(model_xml_path)
        model = mujoco_py.load_model_from_path(model_xml_path)
        n_arms = len(model.actuator_names)
        
        n_arms_str = "{}-collisions_{}_constraints_{}".format(n_arms, collisions, constraints)

        # Identify the model
        method = "{}-collisions_{}_constraints_{}".format(method, collisions, constraints)

    elif env == "Hand":
        model_xml_path = join(gym_assets,  "hand/reach.xml")
        model = mujoco_py.load_model_from_path(model_xml_path)
        n_arms = len(model.actuator_names)
    

    # Experiment identifier if provided
    if identifier:
        method = '{}-{}'.format(method, identifier)

    # Method being a folder will help with organization
    method = "{}/{}".format(method, method)
    if not os.path.isdir(method):
        os.makedirs(method)

    
    #n_arms_str = n_arms
    # Folder to store logs in
    logs = "logs/{}/{}/{}/{}".format(env, n_arms_str or n_arms, hidden, method)
    
    # Storing the logs requires the setting of an environmental variable
    store_logs = "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) \
                        if log else "OPENAI_LOG_FORMAT=stdout"
    
    # Run in parallel on lgns
    parallel  = "mpirun -np 19" if parallel else ""
    num_env   = "--num_env=2" if parallel else ""
    save_path = "models/{}.model".format(method)

    if env == 'Fetch':
        command = """
                   {}
                   {}
		   python -m baselines.run 
                   --alg=her 
                   --env=FetchReachAct-v1 
                   --num_timesteps={} 
                   --n_arms {}
                   {}
                   {} 
                   """.format(store_logs, parallel, num_timesteps, n_arms, num_env, play)
    elif env == "Hand":
        command = """
                   {} 
                   {}
                   python -m baselines.run 
                   --alg=her 
                   --env=HandReach-v0 
                   --num_timesteps={} 
                   --n_arms {}
                   {}
                   {} 
                   """.format(store_logs, parallel, num_timesteps, n_arms, num_env, play)
    else:
        command = """
                   {}
                   {}
                   python -m baselines.run 
                   --alg=her 
                   --env=Arm-v0 
                   --num_timesteps={} 
                   --n_arms {} 
                   {}
                   """.format(store_logs, parallel, num_timesteps, n_arms, play)

    # To a normal looking sentence
    command = " ".join(command.split())
        
    # Add kwargs 
    kwargs['hidden'] = hidden
    kwargs['parts']  = parts
    kwargs['relative_goals'] = relative_goals
    kwargs['normalized'] = normalized
    kwargs['model_xml_path'] = model_xml_path
    
    if not load_path:  
        kwargs['save_path'] = save_path
    else:
        kwargs['load_path'] = save_path

    kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])

    # Kwargs
    command = "{} {}".format(command, kwargs_args)
    print(command)
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
