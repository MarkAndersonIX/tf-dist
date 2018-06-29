#PBS -l walltime=24:00:00
#PBS -o output/infer-${PBS_JOBID}-o.txt
#PBS -e error/infer-${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
source activate tensorflow
python infer.py \
     --image_path=c2.jpg

