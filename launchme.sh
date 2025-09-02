#!/bin/bash

#send to queue with e.g.:
#sbatch --job-name=get_skill_pvpot_enso1 --partition=meteo_long --mem=32G --time=02:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 launchme.sh

#load your software
source ${HOME}/.bashrc

#input parameters
mode='plot' #set Python script to be run. Either 'plot' or 'get_skill'; the latter is depreciated because it has been substituted by <launch_get_skill_season.sh>
log_label='plot_1mon'

#check python version
echo "Your Python version is:"
python --version
#launch to queue on lustre
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG
cd ${RUNDIR}

if [ ${mode} = 'get_skill' ]
then
	python get_skill_season.py > ${LOGDIR}/${log_label}.log
	echo 'Warning: The '${mode}' option is depreciated and will be deactivated in future versions of this script !'
elif [ ${mode} = 'plot' ]
then
	python plot_seasonal_validation_results.py > ${LOGDIR}/${log_label}.log
else
	echo 'Unknown entry mode=${mode}, exiting now....'
fi

echo "launme.sh has been sent to queue successfully, exiting now..."
exit()
