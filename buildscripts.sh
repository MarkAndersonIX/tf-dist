#set error for undefined variable expansion
set -u
#arg parsing: #ps #wk and training script name
pscount=$1
wkcount=$2
train_script=$3
#filenames for ps and worker scripts
psname=_ps
wkname=_wk
#create trainer.sh
> trainer.sh
echo '
> nodes
> joblist
mkdir -p error
mkdir -p output
declare -a jobs=(' >> trainer.sh
	for((i=0;i<$pscount;i+=1)); do echo '"'${psname}${i}'.sh"' >> trainer.sh ;done;
	for((i=0;i<$wkcount;i+=1)); do echo '"'${wkname}${i}'.sh"' >> trainer.sh ;done; echo ')
#output the jobcount to a file
echo "nodecount:" ${#jobs[@]} >> nodes
nodecount=${#jobs[@]}
#qsub each of the jobs and add job ids to list
for i in ${jobs[@]}; do qsub $i | tee -a joblist; done;
#wait for all nodes to allocate
allocated=0
time=0
while [ $allocated -ne $nodecount ]
do
 allocated=`qstat -r | grep "'"R "'" | wc -l`
 echo "waiting for allocated nodes.. time:" $(($time/60))"min"
 sleep 60 
 let "time=time+60"
done
echo "all nodes allocated."' >> trainer.sh

#create ps scripts
for ((i=0;i<$pscount;i+=1))
do
> ${psname}${i}.sh
echo -n '#PBS -l walltime=24:00:00
#PBS -o output/ps'${i}'-${PBS_JOBID}-o.txt
#PBS -e error/ps'${i}'-${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
source activate tensorflow

echo "ps'${i}'" `hostname` >> nodes
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

eval "python '$train_script' \
     --ps_hosts=$pshosts \
     --worker_hosts=$wkhosts \
     --job_name=ps \
     --task_index='${i}'"

' >> ${psname}${i}.sh
#write extra line to file!
done

#create wk scripts
for ((i=0;i<$wkcount;i+=1))
do
> ${wkname}${i}.sh
echo -n '#PBS -l walltime=24:00:00
#PBS -o output/wk'${i}'-${PBS_JOBID}-o.txt
#PBS -e error/wk'${i}'-${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
source activate tensorflow

echo "wk'${i}'" `hostname` >> nodes
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

eval "python '$train_script' \
     --ps_hosts=$pshosts \
     --worker_hosts=$wkhosts \
     --job_name=worker \
     --task_index='${i}'"

' >> ${wkname}${i}.sh
#write extra line to file!
done
chmod +x *.sh
