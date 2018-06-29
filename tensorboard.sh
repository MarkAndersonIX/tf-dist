#PBS -l walltime=24:00:00
#PBS -o output/wk0-${PBS_JOBID}-o.txt
#PBS -e error/wk0-${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
source activate tensorflow
tensorboard --logdir=logs
