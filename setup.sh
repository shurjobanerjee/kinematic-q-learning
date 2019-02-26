module purge
module load cuda/9.0 cudnn/9.0
export PYTHONPATH=""
export FAKETIME=-150d 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1:/usr/lib/x86_64-linux-gnu/libGLEW.so
source env/bin/activate


