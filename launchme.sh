#!/bin/bash

#send to queue with e.g.:
#sbatch --job-name=get_skill_pvpot_enso1 --partition=meteo_long --mem=32G --time=02:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 launchme.sh

#load your software
source ${HOME}/.bashrc

#input parameters
mode='get_skill' #set Python script to be run. Either 'get_skill' or 'plot'

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
