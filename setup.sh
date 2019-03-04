module purge
#module load cuda/9.0 cudnn/9.0
module load cuda/9.0 cudnn/8.0-v7.0.5
export PYTHONPATH=""
export FAKETIME=-150d 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1:/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shurjo/.mujoco/mjpro150/bin 

if [ "$HOSTNAME" = lgn5 ]; then
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-367 
else
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-375
fi
source env/bin/activate


