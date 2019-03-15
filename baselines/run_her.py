import click
import keyword2cmdline
import subprocess
import mujoco_py
import os
from os.path import join, abspath, basename

ENVS = dict(Fetch   = 'FetchReachAct-v1',
            Fetch2D = 'FetchReachAct-v1',
            Hand    = 'HandReach-v0',
            Arm     = 'Arm-v0',
           )


#echo "allocated node"; cat $PBS_NODEFILE
#echo "GPU allocated"; cat $PBS_GPUFILE
def makedirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

@keyword2cmdline.command
def main(num_timesteps=5000, play=True, parts='None', n_arms=2, env='Arm',
        hidden=16, identifier='', normalized=False, parallel=False,
        save_path=False,  constraints=False, collisions=False, 
        relative_goals=True, qsub=False, seed=0, **kwargs):
    
    
    # Constant hyperparams 
    mtype = "rel_goals-{}-hidden-{}".format(relative_goals, hidden)
    
    # Differentiate the method
    method = 'baseline' if parts is 'None' else 'ours'
    if normalized:
        method = '{}-normalized={}'.format(method, normalized)
    if identifier:
        method = '{}={}'.format(method, identifier)
    method = '{}-{}'.format(method, seed)

    # Environment specifics
    env_name = ENVS[env]

    # Model file name changes based on params
    model_xml_path = None
    gym_assets = "../gym/gym/envs/robotics/assets"
    if 'Fetch' in env:
        if env == 'Fetch2D':
            # Arms constrained to a plane
            xml_name = "fetch/reach-actuated.2d.collisions_{}_constraints_{}.xml".format(\
                                 collisions, constraints)
            mtype = "{}-collisions={}_constraints={}".format(mtype, collisions, constraints)
        else:
            # Raw Fetch experiment
            xml_name = "fetch/reach-actuated.xml"
        
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
        # 2D arm simulated (how many arms are used)
        env = "{}-{}".format(env, n_arms)

    # Experiment identifier if provided

    # Folder to store logs in
    logs = "logs/{}/{}/{}".format(env, mtype, method)
    makedirs(logs)
    
    # Run in parallel on lgns
    parallel  = "mpirun -np 19" if parallel else ""
    num_env   = "--num_env=2" if parallel else ""

    # Save the model to the log folder
    save_path = "{}/{}.model".format(logs, method)
    
    # Format keyword arguments for running HER
    kwargs['seed'] = seed
    kwargs['env'] = env_name
    kwargs['num_timesteps'] = num_timesteps
    kwargs['n_arms'] = n_arms
    kwargs['hidden'] = hidden
    kwargs['parts']  = parts
    kwargs['relative_goals'] = relative_goals
    kwargs['normalized'] = normalized
    kwargs['model_xml_path'] = model_xml_path
    kwargs['save_path'] = save_path
    
    # To a normal looking sentence
    train_command = make_command(logs, parallel, **kwargs)
    # Run the code locally or on the cluster
    print(train_command,'\n')
    train = make_executable_script(logs, train_command, True)
    subprocess.call('{}{}'.format('qsub ' if qsub else './', train), shell=True)
    
    # Save test command to visualize on retinene later
    kwargs.pop('save_path')
    kwargs['load_path'] = save_path + " --play"
    kwargs['num_timesteps'] = 1

    # To a normal looking sentence
    test_command = make_command(logs, parallel, **kwargs)
    test = make_executable_script(logs, test_command, False)
    #if not qsub:
    #    subprocess.call('./{}'.format(test), shell=True,
    #                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_command(logs, parallel, **kwargs):
    kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])
    
    # Log only training
    logs =  "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if logs else ""

    # Command to run 
    command = "DISPLAY=:0 {} {} python -m baselines.run --alg=her {}".format(\
            logs, parallel, kwargs_args)

    # Save the plots

    return " ".join(command.split())

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

if __name__ == "__main__":
    main()
