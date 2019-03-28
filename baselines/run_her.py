import click
import keyword2cmdline
import subprocess
import mujoco_py
import os
from os.path import join, abspath, basename
from jinja2 import Template
from keyword2cmdline import opts
import sys
import socket

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
            format_str(normalized=normalized), seed)
    return method

def get_log_dir(env, hyper_param_str, method, n_arms):
    # Folder to store logs in
    if False: #socket.gethostname() == "retinene":
        log_folder = "logs-retinene"
    else:
        log_folder = "logs"
    logs = join(log_folder, env, str(n_arms), hyper_param_str, method)
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



PBS="""#!/bin/bash
#PBS -N {jobname}             # Any name to identify your job
#PBS -j oe                    # Join error and output files for convinience
#PBS -l walltime=200:00:00    # Keep walltime big enough to finish the job
#PBS -l nodes=1:ppn=10 # nodes requested: Processor per node: gpus requested
#PBS -S /bin/bash             # Shell to use
#PBS -m abe                   # Mail to <user>@umich.edu on abort, begin and end
#PBS -V                       # Pass current environment variables to the script
#PBS -e pbs_output
#PBS -o pbs_output

#echo "allocated node"; cat $PBS_NODEFILE

python -c "import socket; print(socket.gethostname())"

cd {cwd}

#if [ ! -f ~/.mujoco ]; then
#    cp -R .mujoco.dhiman/ ~/.mujoco 
#fi
"""


def make_executable_script(logs='', command='', train=True, mode='human'):
    pbs = PBS.format(jobname=basename(logs), pbs_output=logs, cwd=os.getcwd())
    sourcer = 'source {} \n'.format(abspath('../setup.sh'))
    if mode == 'human':
        sourcer += "export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so\n"

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
        relative_goals=True, qsub=False, seed=0, stdout=sys.stdout, stderr=sys.stderr,
        mode='rgb_array', **kwargs):
    
    # Organize experiments by hyperparams 
    hyper_param_str = get_hyper_param_str(env, collisions,
                            constraints=constraints, relative_goals=relative_goals,
                            hidden=hidden)
    if identifier: 
        hyper_param_str = "{}_{}".format(hyper_param_str, identifier)

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
                  model_xml_path=model_xml_path, collisions=collisions,
                  constraints=constraints, mode=mode, play=play)

    # Training
    kwargs['save_path'] = save_path
    train_command = make_command(logs, True, **kwargs)
    train = make_executable_script(logs, train_command, True, mode)
    subprocess.call('{}{}'.format('qsub ' if qsub else './', train), shell=True,
                        stdout=stdout, stderr=stderr)

    # Save test command for model-based visualization if required
    kwargs.pop('save_path')
    kwargs['load_path'] = save_path

    # Testing
    kwargs['num_timesteps'] = 1
    test_command = make_command(logs, False, **kwargs)
    test = make_executable_script(logs, test_command, False, mode)
    #if not qsub:
    #    subprocess.call('./{}'.format(test), shell=True,
    #                    stdout=stdout, stderr=stderr)


def make_command(logs, train, parallel=False, play=False, **kwargs):
    kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])
    
    # Log only training
    logs =  "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if train else ""

    # Run in parallel on lgns
    mpi, num_env  = ("mpirun -np 19", "--num_env=2") if parallel else ("","")

    # Play command
    play = "--play" if play else ""
    
    # Command to run 
    command = "{} {} python -m baselines.run --alg=her {} {} {}".format(\
            logs, mpi, kwargs_args, num_env, play)

    return command


@keyword2cmdline.command
def main(num_timesteps=15000, play=True, parts='None', n_arms=2, env='Arm',
        hidden=16, identifier='', normalized=False, parallel=False,
        save_path=False,  constraints=False, collisions=False, 
        relative_goals=True, qsub=False, seeds="0", debug=False, 
        verbose = True, **kwargs):
    
    seeds = list(map(int, seeds.split(',')))
    normalizeds = [True, False]
    partss = ['diff', 'None']
    if debug:
        seeds, normalizeds, partss = map(lambda x: x[0:1], [seeds, normalizeds, partss])
    
    if not verbose:
        stdout = sys.stdout
        null = open(os.devnull, 'w')

    counter = 1
    total   = len(seeds) * len(normalizeds) * len(partss)
    for seed in seeds:
        for normalized in normalizeds:
            for parts in partss:
                if not verbose:
                    sys.stdout = stdout
                    sys.stderr = stdout
                    print("{}/{}".format(counter, total))
                    counter += 1
                    sys.stdout = null
                    sys.stderr = null

                # Runs locally or via qsub
                run_her(num_timesteps, play, parts, n_arms, env,
                        hidden, identifier, normalized, parallel,
                        save_path,  constraints, collisions, 
                        relative_goals, qsub, seed, stdout=sys.stdout, 
                        stderr=sys.stderr, **kwargs)
if __name__ == "__main__":
    main()
