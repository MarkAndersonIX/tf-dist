if [ $# -ne 0 ] && [ ${1} == "all" ]
then
	rm _ps*
	rm _wk*
	rm trainer.sh
	rm -rf logs/*
fi
	rm -rf output/*
	rm -rf error/*
