#!/bin/bash

####################################################################################################
# get_skill_season.sh:
####################################################################################################
#
# Description: This script is launched to queue with send2queue_get_skill_seasons.sh. Executed on
# a working node, it launches the Python script <get_skill_season.py>, which accomplishes the model
# verification against observations, calculates the quantiles and the tercile probability time-series,
# and finally stores all the results in netCDF format. Separate files are generated for each tempora
# aggregation window (1 to 5 months), model and variable. <get_skill_season.py> is designed like this
# on purpose in order to send small and short jobs to queue that accomplish the aformentioned tasks in
# a modular manner.
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#load your envirmonent
source ${HOME}/.bashrc
sleep 3

#input variables that will be passed to the python script get_skill_season.py
vers=${1}
model=${2}
domain=${3}
variable=${4}
agg_label=${5}
modulator=${6}
phase=${7}
RUNDIR=${8}
PYDIR=${9}
LOGDIR=${10}
FLAGDIR=${11}
jobname=${12}

## EXECUTE #########################################################################
#check python version
echo "Your Python version is:"
python --version

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}
# current_time=$(date "+%Y-%m-%d_%H-%M-%S")

log_label=get_skill_season_py_${jobname}
echo "The log_label sent to get_skill_season.py is "${log_label}
# run the Python script

python ${PYDIR}/get_skill_season.py ${vers} ${model} ${domain} ${variable} ${agg_label} ${modulator} ${phase} ${FLAGDIR} > ${LOGDIR}/${log_label}.log

echo "get_skill_season.sh has been sent to queue successfully with the parameters vers: ${vers}, model: ${model}, domain: ${domain}, variable: ${variable}, agg_label: ${agg_label}, modulator: ${modulator}, phase: ${phase}, FLAGDIR: ${FLAGDIR}, LOGDIR: ${LOGDIR}, log_label: ${log_label}."
echo "Exiting now..."
sleep 1
exit 0
