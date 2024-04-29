#!/bin/bash

#send to queue with e.g.:
#qsub -N get_skill -l walltime=05:00:00 -l mem=16gb -q himem -e error.log -o out.log -l nodes=1:ppn=1 launchme.sh
#load your software
source ${HOME}/.bashrc

#input parameters
mode='plot' #set Python script to be run. Either 'get_skill' or 'plot'

#check python version
echo "Your Python version is:"
python --version
#launch to queue on lustre
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG
cd ${RUNDIR}

if [ ${mode} = 'get_skill' ]
then
	python get_skill_season.py > ${LOGDIR}/log_get_skill_season.log
elif [ ${mode} = 'plot' ]
then
	python plot_seasonal_validation_results.py > ${LOGDIR}/log_plot_seasonal_validation_results.log
else
	echo 'Unknown entry mode=${mode}, exiting now....'
fi

echo "launme.sh has been sent to queue successfully, exiting now..."
exit()
