import click
import keyword2cmdline
import subprocess

@keyword2cmdline.command
def main(num_timesteps=5000, play=False, log=True, parts=False, n_arms=2, env='2d'):
    play = "" if not play else "--play"
    logs = "logs/{}/{}/{}".format(env, n_arms, 'baseline' if not parts else 'ours')
    store_logs = "OPENAI_LOGDIR={} OPENAI_LOG_FORMAT=csv,stdout".format(logs) if log else "OPENAI_LOG_FORMAT=stdout"

    if env == 'Arm':
        command = "{} python -m baselines.run --alg=her --env=FetchReachAct-v1 --num_timesteps={} {}".format(store_logs, num_timesteps, play)
    else:
        command = "{} python -m baselines.run --alg=her --env=Arm-v0 --num_timesteps={} --n_arms {} --parts {} {}".format(store_logs, num_timesteps, n_arms, parts, play)
        print(command)
    subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
