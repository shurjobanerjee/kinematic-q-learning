from baselines.common import plot_util as pu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keyword2cmdline
from os.path import join, basename
import glob

def load_data(method, ttype, smooth=False):
    try:
        plt.clf()
        results = pu.load_results(method)
        r = results[0]
        eps  = r[1].epoch.values #np.cumsum(r[1].epoch).values
        dist = r[1]['{}/success_rate'.format(ttype)].values 
        if smooth:
            dist = pu.smooth(dist, radius=2)
        # Starting index
        start = 1 if np.isnan(dist[0]) else 0
        return eps[start:], dist[start:]
    except Exception as e:
        print(method, str(e))
        return [],[]

#@keyword2cmdline.command
def plot(logs='logs/Arm/', ttype='test', smooth=False):
    logs = glob.glob(join(logs, '*'))
    for log in logs:
        for ttype in ['train', 'test']:
            data = dict(ours=[], baseline=[])
            methods = [join(log, x) for x in ['baseline', 'ours']]
            for method in methods:
                data[basename(method)] = load_data(method, ttype)
            
            # FIXME Do this with 'hold on' like behavior
            plt.plot(data['baseline'][0], data['baseline'][1],
                     data['ours'][0], data['ours'][1],)
            plt.legend(['baseline', 'ours'])
            plt.title(method)
            plt.savefig('{}/{}.png'.format(log,ttype))


if __name__ == "__main__":
    plot()
