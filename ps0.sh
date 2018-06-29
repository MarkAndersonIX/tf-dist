#PBS -l walltime=24:00:00
#PBS -o output/ps0-${PBS_JOBID}-o.txt
#PBS -e error/ps0-${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
source activate tensorflow

echo "ps0" `hostname` >> nodes
nodecount=$(head -n 1 nodes | sed "s/nodecount: //")
while [ `cat nodes | grep -v "nodecount" | wc -l` -lt $nodecount ]
do
	sleep 60
done
cat nodes
pshosts=$(cat nodes | grep "ps" | sort | sed "s/.* //")
wkhosts=$(cat nodes | grep "wk" | sort | sed "s/.* //")
pshosts=( ${pshosts[@]} )
wkhosts=( ${wkhosts[@]} )

function join { local IFS="$1"; shift; echo "$*"; }

pshosts=`join , ${pshosts[@]/%/:2222}`
wkhosts=`join , ${wkhosts[@]/%/:2222}`
echo $pshosts
echo $wkhosts

eval "python trainer.py \
     --ps_hosts=$pshosts \
     --worker_hosts=$wkhosts \
     --job_name=ps \
     --task_index=0"

