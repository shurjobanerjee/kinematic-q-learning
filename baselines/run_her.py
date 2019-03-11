import click
import keyword2cmdline
import subprocess
import mujoco_py
import os
from os.path import join, abspath

ENVS = dict(Fetch   = 'FetchReachAct-v1',
            Fetch2D = 'FetchReachAct-v1',
            Hand    = 'HandReach-v0',
            Arm     = 'Arm-v0',
           )


def makedirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

@keyword2cmdline.command
def main(num_timesteps=5000, play=True, parts='None', n_arms=2, env='Arm',
        hidden=16, identifier='', normalized=False, parallel=False,
        save_path=False,  constraints=False, collisions=False, load_path=False,
        relative_goals=True, **kwargs):
    
    # Governs whether to show a test simulation
    play = "" if not play else "--play"

    # Differentiate the method
    method = 'baseline' if parts is 'None' else 'our'
    
    # Constant hyperparams 
    method = "{}-rel_goals-{}-normalized-{}-hidden-{}".format(method, relative_goals, normalized, hidden)

    # Model file name changes based on params
    model_xml_path = None
    gym_assets = "../gym/gym/envs/robotics/assets"
    if 'Fetch' in env:
        if env == 'Fetch2D':
            # Arms constrained to a plane
            xml_name = "fetch/reach-actuated.2d.collisions_{}_constraints_{}.xml".format(\
                                 collisions, constraints)
        else:
            # Raw Fetch experiment
            xml_name = "fetch/reach-actuated.xml"
            method = "{}-collisions_{}_constraints_{}".format(method, collisions, constraints)
        
        # Formulate the model xml path that is loaded for MujoCo
        model_xml_path = join(gym_assets, xml_name)
        model_xml_path = abspath(model_xml_path)
        model = mujoco_py.load_model_from_path(model_xml_path)
        n_arms = len(model.actuator_names)

    elif env == "Hand":
        # Hand Environment
        model_xml_path = join(gym_assets,  "hand/reach.xml")
        model = mujoco_py.load_model_from_path(model_xml_path)
        n_arms = len(model.actuator_names)

    else:
        # 2D arm simulated
        pass

    # Experiment identifier if provided
    if identifier:
        method = '{}-{}'.format(method, identifier)

    # Folder to store logs in
    logs = "logs/{}/{}".format(env, method)
    makedirs(logs)
    
    # Run in parallel on lgns
    parallel  = "mpirun -np 19" if parallel else ""
    num_env   = "--num_env=2" if parallel else ""

    # Save the model to the log folder
    save_path = "{}/{}.model".format(logs, method)

    # Environment specifics
    env = ENVS[env]
    
    # Format keyword arguments for running HER
    kwargs['env'] = env
    kwargs['num_timesteps'] = num_timesteps
    kwargs['n_arms'] = n_arms
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
    
    
    # Run the code
    command = """
               DISPLAY=:0
               OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout
               {}
               python -m baselines.run 
               --alg=her 
               {}
               {}
               """.format(logs, parallel, kwargs_args, play)


    # To a normal looking sentence
    command = " ".join(command.split())
        
    # Run the code locally or on the cluster
    print(command,'\n')
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
