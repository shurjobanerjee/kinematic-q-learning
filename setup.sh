module purge
#module load cuda/9.0 cudnn/9.0
module load cuda/9.0 cudnn/8.0-v7.0.5
export PYTHONPATH=""
export FAKETIME=-210d 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shurjo/.mujoco/mjpro150/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(ls -d /usr/lib/nvidia-???/ | sort -r | head -1)
export DISPLAY=:0
source env/bin/activate


