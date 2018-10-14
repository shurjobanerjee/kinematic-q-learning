import click
import keyword2cmdline
import subprocess

class Params:
    parts = False
    n_arms = 3
    etype = '2d' #'Arm' 

@keyword2cmdline.command
def main(num_timesteps=5000, play=False, log=True):
    play = "" if not play else "--play"
    logs = "logs-arm/{}/{}".format(Params.n_arms, 'baseline' if not Params.parts else 'ours')
    store_logs = "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if log else "OPENAI_LOG_FORMAT=stdout"

    if Params.etype == 'Arm':
        command = "{} python -m baselines.run --alg=her --env=FetchReachAct-v1 --num_timesteps={} {}".format(store_logs, num_timesteps, play)
    else:
        command = "{} python -m baselines.run --alg=her --env=Arm-v0 --num_timesteps={} {} --mode rgb_array".format(store_logs, num_timesteps, play)
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
