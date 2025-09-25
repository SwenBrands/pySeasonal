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

#input variables that will be passed to the python script get_skill_season.py
vers=$1
model=$2
variable=$3
agg_label=$4
modulator=$5
phase=$6
RUNDIR=$7
LOGDIR=$8
FLAGDIR=$9
jobname=$10

## EXECUTE #########################################################################
log_label=get_skill_season_${jobname}
#check python version
echo "Your Python version is:"
python --version

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}
# current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# run the Python script
python get_skill_season.py ${vers} ${model} ${variable} ${agg_label} ${modulator} ${phase} ${FLAGDIR} > ${LOGDIR}/${log_label}.log

echo "get_skill_season.sh has been sent to queue successfully, exiting now..."
sleep 1
exit 0
