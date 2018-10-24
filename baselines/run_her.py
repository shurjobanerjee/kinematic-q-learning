import click
import keyword2cmdline
import subprocess

@keyword2cmdline.command
def main(num_timesteps=5000, play=False, log=True, parts=False, n_arms=2, env='2d', conn_type='sum', relative_goals=False, **kwargs):
    play = "" if not play else "--play"
    method = 'baseline' if not parts else 'ours-{}'.format(conn_type)
    method = "{}-rel_goals-{}".format(method, relative_goals)
    logs = "logs/{}/{}/{}".format(env, n_arms, method)
    store_logs = "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if log else "OPENAI_LOG_FORMAT=stdout"

    if env == 'Arm':
        command = "{} python -m baselines.run --alg=her --env=FetchReachAct-v1 --num_timesteps={} {}".format(store_logs, num_timesteps, play)
    else:
        
        if kwargs:
            kwargs_args = ' '.join(['--{} {}'.format(k,f) for k, f in kwargs.items()])
        else:
            kwargs_args = ''

        command = "{} python -m baselines.run --alg=her --env=Arm-v0 --num_timesteps={} --n_arms {} --parts {} --conn_type {} relative_goals {} {} {}".format(store_logs, num_timesteps, n_arms, parts, conn_type, relative_goals, kwargs_args, play)
        print(command)
    
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
