
> nodes
> joblist
mkdir -p error
mkdir -p output
declare -a jobs=(
"ps0.sh"
"wk0.sh"
"wk1.sh"
"wk2.sh"
"wk3.sh"
)
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
 allocated=`qstat -r | grep "R " | wc -l`
 echo "waiting for allocated nodes.. time:" $(($time/60))"min"
 sleep 60 
 let "time=time+60"
done
echo "all nodes allocated."
