from baselines.common import plot_util as pu
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keyword2cmdline
import os
from os.path import join, basename, abspath, dirname
import glob
import shutil
def makedirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

def ep_distance_ratio_train(r):
    x = r[1].epoch
    y = r[1]['train/success_rate']
    x, y = map(np.asarray, (x, y))
    #y = pu.smooth(y, radius=10)
    start = 1 if np.isnan(y[0]) else 0
    return x[start:],y[start:]

def ep_distance_ratio_test(r):
    x = r[1].epoch
    y = r[1]['test/success_rate']
    x, y = map(np.asarray, (x, y))
    #y = pu.smooth(y, radius=10)
    start = 1 if np.isnan(y[0]) else 0
    return x[start:],y[start:]

def cglob(path):
    """
    Custom glob that ignores pngs
    """
    files = glob.glob(join(path, '*'))
    files = [f for f in files if 'png' not in f]
    return files

def organize_results(results):
    """Order results so that same methods have same colors"""
    ndxs = np.argsort([r.dirname for r in results])
    return [results[n] for n in ndxs]


def plot_data(exp, savefig, ttype):
    savefig = abspath(savefig)
    try:
        results = organize_results(pu.load_results(exp))
        pu.plot_results(results, 
                        average_group=True, 
                        split_fn=lambda _: '',
                        xy_fn=ep_distance_ratio_train \
                                if ttype == 'train' else ep_distance_ratio_test,
                        shaded_std=False,
                        shaded_err=True)
        if os.path.isfile(savefig): os.remove(savefig)
        plt.savefig(savefig)
        plt.clf()
        #print("Plot saved to: {}".format(savefig))
    except Exception as e:
        print("Plotting failed for {}".format(savefig))
        print("Reason: {}".format(str(e)))

def plot(logs='logs', ttype='test', smooth=False):
    envs = cglob(logs)
    for env in envs:
        arms = cglob(env)
        for arm in arms:
            exps = cglob(arm)
            for exp in exps:
                for ttype in ['train', 'test']:
                    plot_data(exp, '{}/{}.png'.format(exp, ttype), ttype)


    ## Plot in one giant plot (easy comparison)
    #tmp_plots = '/tmp/plots/'
    #shutil.rmtree(tmp_plots)
    #makedirs(tmp_plots)

    #for env in envs:
    #    arms = cglob(env)
    #    for arm in arms:
    #        exps = cglob(arm)
    #        arm_dir = join(tmp_plots, basename(env), basename(arm))
    #        makedirs(arm_dir)
    #        for exp in exps:
    #            methods = cglob(exp)
    #            for method in methods:
    #                new_dir = join(arm_dir, basename(exp) + '_' + basename(method))
    #                #os.symlink(abspath(method), abspath(new_dir))
    #                shutil.copytree(abspath(method), abspath(new_dir))
    #        
    #        import pdb; pdb.set_trace()
    #        for ttype in ['train', 'test']:
    #            plot_data(arm_dir, '{}/{}.png'.format(arm_dir, ttype), ttype)
    #
    #import pdb; pdb.set_trace()


def make_intuitive(filename):
    return filename.replace('/','-').replace('train.png','0-train.png').replace('test', '1-test')

def rmtree(folder):
    if os.path.isdir(folder):
        for f in glob.glob(join(folder, '*')):
            os.remove(f)

def organize(logs='logs', results='results'):
    plots     = glob.glob(join(logs, '**', '*.png'), recursive=True)
    plots_sym = [join(results, make_intuitive(p)) for p in plots]
    rmtree(results)
    makedirs(results)
    for p, ps in zip(plots, plots_sym):
        os.symlink(abspath(p), abspath(ps))

if __name__ == "__main__":
    plot()
    organize()
