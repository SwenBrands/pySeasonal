#!/bin/bash


####################################################################################################
# launch_get_skill_season.sh:
####################################################################################################
#
# Description: This script is launched to queue with launch_get_skill_seasons.sh. Itsefl, it launches the Python script get_skill_season.py, which accomplishes the model verification against observations and stores the results in netCDF format
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#load your envirmonent
source ${HOME}/.bashrc

#input variables that will be passed to the python script get_skill_season.py
vers=$1
model=$2
agg_label=$3
modulator=$4
phase=$5
RUNDIR=$6
LOGDIR=$7

## EXECUTE #########################################################################
log_label=get_skill_season_${vers}_${model}_${agg_label}_${modulator}_${phase}
#check python version
echo "Your Python version is:"
python --version

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}
# current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# run the Python script
python get_skill_season.py ${vers} ${model} ${agg_label} ${modulator} ${phase}> ${LOGDIR}/${log_label}.log

echo "get_skill_season.sh has been sent to queue successfully, exiting now..."
sleep 1
exit 0
