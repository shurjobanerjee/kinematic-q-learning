from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import keyword2cmdline
from os.path import join, basename
import glob

@keyword2cmdline.command
def main(logs='logs-arm', n_arms=9, smooth=False, ttype='test'):
    print(logs)
    logs = glob.glob(join(logs, str(n_arms),'*'))
    print(logs)
    for log in logs:
        print(log)
        try:
            print(log)
            results = pu.load_results(log)
            r = results[0]
            eps  = np.cumsum(r[1].epoch).values
            dist = r[1]['{}/success_rate'.format(ttype)].values 
            if smooth:
                dist = pu.smooth(dist, radius=2)
            plt.plot(eps, dist, label=basename(log))
        except Exception as e:
            print(str(e))
    plt.legend()
    plt.title('Number of arms: {}'.format(n_arms))
    plt.show()


if __name__ == "__main__":
    main()
