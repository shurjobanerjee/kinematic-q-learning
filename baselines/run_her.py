import click
import keyword2cmdline
import subprocess
import mujoco_py
import os
from os.path import join, abspath, basename
from jinja2 import Template
from keyword2cmdline import opts

ENVS = dict(Fetch   = 'FetchReachAct-v1',
            Fetch2D = 'FetchReachAct-v1',
            Hand    = 'HandReach-v0',
            Arm     = 'Arm-v0',
           )

def makedirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def load_xml_template(filename, **kwargs):
    with open(filename) as file_:
        template = Template(file_.read())
    return template.render(**kwargs)

def get_hyper_param_str(env, collisions=False, **kwargs):
    # Sort kwargs for consistency
    kwargs = dict(sorted(kwargs.items()))
    # Collsiions/constraints placed at the end cause of Ordered Dict
    if env != "Arm":
        kwargs.update(collisions=collisions)
    return format_str(**kwargs)

def format_str(delimiter="=", **kwargs):
    output = ["{}{}{}".format(k, delimiter, v) for k,v in kwargs.items() if v != '']
    return '_'.join(output)

def get_method_name(parts, normalized, identifier, seed):
    """Name of method applied"""
    # Differentiate the method
    method = 'baseline' if parts is 'None' else 'ours'
    method = "{}_{}-{}".format(method, \
            format_str(normalized=normalized, identifier=identifier), seed)
    return method

def get_log_dir(env, hyper_param_str, method, n_arms):
    # Folder to store logs in
    logs = join("logs", env, str(n_arms), hyper_param_str, method)
    makedirs(logs)
    return logs


def save_xml_file(env, collisions, constraints, n_arms, **kwargs):
    # Save file location
    save_to = "../gym/gym/envs/robotics/assets/fetch/"
    fname_ext = format_str(constraints=constraints, collisions=collisions,
                           n_arms=n_arms, delimiter='_')

    # Shared file (collisions)
    shared_filename = 'sim_templates/fetch/shared.xml.jinja'
    shared = load_xml_template(filename=shared_filename, **kwargs)
    new_shared_filename = "shared_{}.xml".format(fname_ext)
    new_shared_filename = join(save_to, new_shared_filename)
    with open(new_shared_filename, 'w') as f:
        f.write(shared)

    # Specific file (constraints)
    robot_filename = 'sim_templates/fetch/reach-actuated.xml.jinja'
    robot = load_xml_template(filename=robot_filename, 
                              shared=basename(new_shared_filename), 
                              constraints=str(constraints).lower(),
                              n_arms=n_arms)
    new_robot_filename = "reach-actuated_{}.xml".format(fname_ext)
    new_robot_filename = join(save_to, new_robot_filename)
    with open(new_robot_filename, 'w') as f:
        f.write(robot)
    
    return abspath(new_robot_filename)



def get_model_xml_path(env, collisions, constraints, n_arms):
    ## Load Jinja templates
    #model_file = load_xml_template(filename='sim_templates/reach-actuated.xml.jinja')
    # Model file name changes based on params
    if env == 'Arm':
        return None
    else:
        if 'Fetch' in env:
            # Make the shared file
            model_xml_path = save_xml_file(env, \
                    collisions=collisions, 
                    constraints=constraints, 
                    n_arms=n_arms)
            print(model_xml_path)
            model = mujoco_py.load_model_from_path(model_xml_path)

        elif env == "Hand":
            # Hand Environment
            gym_assets = "../gym/gym/envs/robotics/assets"
            model_xml_path = join(gym_assets,  "hand/reach.xml")
            model = mujoco_py.load_model_from_path(model_xml_path)
            n_arms = len(model.actuator_names)
    
        return model_xml_path


def make_command(logs, train, parallel=False, **kwargs):
    kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])
    
    # Log only training
    logs =  "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if train else ""

    # Run in parallel on lgns
    mpi, num_env  = ("mpirun -np 19", "--num_env=2") if parallel else ("","")
    
    # Command to run 
    command = "DISPLAY=:0 {} {} python -m baselines.run --alg=her {}".format(\
            logs, mpi, kwargs_args, num_env)

    return command

PBS="""#!/bin/bash
#PBS -N {jobname}             # Any name to identify your job
#PBS -j oe                    # Join error and output files for convinience
#PBS -l walltime=200:00:00    # Keep walltime big enough to finish the job
#PBS -l nodes=1:ppn=10:gpus=1 # nodes requested: Processor per node: gpus requested
#PBS -S /bin/bash             # Shell to use
#PBS -m abe                   # Mail to <user>@umich.edu on abort, begin and end
#PBS -V                       # Pass current environment variables to the script
#PBS -e {pbs_output}
#PBS -o {pbs_output}

#echo "allocated node"; cat $PBS_NODEFILE

python -c "import socket; print(socket.gethostname())"

cd {cwd}

if [ ! -f ~/.mujoco ]; then
    cp -R .mujoco.dhiman/ ~/.mujoco 
fi
"""


def make_executable_script(logs='', command='', train=True):
    pbs = PBS.format(jobname=basename(logs), pbs_output=logs, cwd=os.getcwd())
    sourcer = 'source {} \n'.format(abspath('../setup.sh'))
    fname = join(logs, 'train.sh' if train else 'test.sh')
    with open(fname, 'w') as f:
        f.write(pbs)
        f.write(sourcer)
        f.write(command)
    subprocess.call("chmod +x {}".format(fname), shell=True)

    return fname

def run_her(num_timesteps=5000, play=True, parts='None', n_arms=2, env='Arm',
        hidden=16, identifier='', normalized=False, parallel=False,
        save_path=False,  constraints=False, collisions=False, 
        relative_goals=True, qsub=False, seed=0, **kwargs):
    
    # Organize experiments by hyperparams 
    hyper_param_str = get_hyper_param_str(env, collisions,
                            constraints=constraints, relative_goals=relative_goals,
                            hidden=hidden)
    # Get the method name
    method = get_method_name(parts, normalized, identifier, seed)
    # Get the directory to store results
    logs = get_log_dir(env, hyper_param_str, method, n_arms)
    # Model file
    model_xml_path = get_model_xml_path(env, collisions, constraints, n_arms)
    # Save the model to the log folder
    save_path = "{}/{}.model".format(logs, method)
    
    
    # Format keyword arguments for running HER
    kwargs.update(seed=seed, env=ENVS[env], num_timesteps=num_timesteps,
                  n_arms=n_arms, hidden=hidden, parts=parts,
                  relative_goals=relative_goals, normalized=normalized,
                  model_xml_path=model_xml_path, save_path=save_path,
                  collisions=collisions, constraints=constraints)
    # Run the code locally or on the cluster
    train_command = make_command(logs, True, **kwargs)
    print(train_command,'\n')
    train = make_executable_script(logs, train_command, True)
    subprocess.call('{}{}'.format('qsub ' if qsub else './', train), shell=True)
    
    
    # Save test command to visualize on retinene later
    kwargs.pop('save_path')
    kwargs['load_path'] = save_path + " --play"
    kwargs['num_timesteps'] = 1
    test_command = make_command(logs, False, **kwargs)
    test = make_executable_script(logs, test_command, False)
    if not qsub:
        subprocess.call('./{}'.format(test), shell=True)
                        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




@keyword2cmdline.command
def main(num_timesteps=15000, play=True, parts='None', n_arms=2, env='Arm',
        hidden=16, identifier='', normalized=False, parallel=False,
        save_path=False,  constraints=False, collisions=False, 
        relative_goals=True, qsub=False, seeds="1,2,3,4,5", debug=False, **kwargs):
    
    if not debug: 
        seeds = map(int, seeds.split(','))
        normalizeds = [True, False]
        partss = ['diff', 'None']
    else:
        seeds = [0]
        normalizeds = [True]
        partss = ['diff']
    
    for seed in seeds:
        for normalized in normalizeds:
            for parts in partss:
                # Runs locally or via qsub
                run_her(num_timesteps, play, parts, n_arms, env,
                        hidden, identifier, normalized, parallel,
                        save_path,  constraints, collisions, 
                        relative_goals, qsub, seed, **kwargs)
if __name__ == "__main__":
    main()
