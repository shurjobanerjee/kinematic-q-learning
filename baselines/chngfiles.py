import glob
import os

files = glob.glob('logs/**/baseline-normalized*', recursive=True)
files = [f for f in files if '.model' not in f]

for f in files:
    os.rename(f, f.replace('baseline-normalized', 'baseline=normalized'))


files = glob.glob('logs/**/ours-normalized*', recursive=True)
files = [f for f in files if '.model' not in f]

for f in files:
    os.rename(f, f.replace('ours-normalized', 'ours=normalized'))
