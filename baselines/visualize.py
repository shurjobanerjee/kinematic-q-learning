import subprocess
import glob
from os.path import join

def main(logs='logs/Hand/'):
    test_files = glob.glob(join(logs, '**', 'test.sh'), recursive=True)
    test_files = [tf for tf in test_files if '-1' in tf]
    for tf in test_files:
        subprocess.call(tf, shell=True)


if __name__ == "__main__":
    main()

