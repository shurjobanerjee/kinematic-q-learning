from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import keyword2cmdline
from os.path import join, basename
import glob

@keyword2cmdline.command
def main(logs='logs-arm', n_arm=3, smooth=True):
    logs = glob.glob(join(logs, str(n_arm),'*'))
    for log in logs:
        results = pu.load_results(log)
        r = results[0]
        eps  = np.cumsum(r[1].epoch).values
        dist = r[1]['test/success_rate'].values 
        if smooth:
            dist = pu.smooth(dist, radius=2)
        plt.plot(eps, dist, label=basename(log))
    plt.legend()
    plt.title('Number of arms: {}'.format(n_arm))
    plt.show()


if __name__ == "__main__":
    main()
