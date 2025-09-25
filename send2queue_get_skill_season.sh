#!/bin/bash

####################################################################################################
# send2queue_get_skill_season.sh:
####################################################################################################
#
# Description: This script is sends get_skill_seasons.sh to a working node and monitors its execution
# the frontal node, writing into a logfile until get_skill_season.py writes a flag after successful execution.
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#load your envirmonent
source ${HOME}/.bashrc
#check python version
echo "Your Python version is:"
python --version

#input variables that will be passed to the python script get_skill_season.py
partition=${1}
exclude_node=${2}
exectime=${3}
memory=${4}
vers=${5}
model=${6}
variable=${7}
agg_label=${8}
modulator=${9}
phase=${10}
RUNDIR=${11}
LOGDIR=${12}
FLAGDIR=${13}
jobname=${14}

## EXECUTE #########################################################################
checktime=60 #time in seconds that passed from one flag check to another

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}

#construct the command to be sent to queue
QSUB="sbatch \
    --partition=${partition}
    --exclude=${exclude_node}
    --time=${exectime} \
    --job-name=${jobname}\
    --export=ALL \
    --begin=now \
    --output=${LOGDIR}/${jobname}.out \
    --ntasks-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=${memory} \
    --mail-user=swen.brands@gmail.com \
    --mail-type=all \
    ./get_skill_season.sh ${vers} ${model} ${variable} ${agg_label} ${modulator} ${phase} ${RUNDIR} ${LOGDIR} ${FLAGDIR} ${jobname}" #get_skill_season.sh contains the Python script to be run on the working node
echo ${QSUB} #prints the command sent to queue
${QSUB} #sent to queue !

#--------------------------------------------------------------------------------------------------
# Check whether get_skill_season.py executes correctly and writes the flag
#--------------------------------------------------------------------------------------------------
flagfile=${FLAGDIR}/get_skill_season_${vers}_${model}_obs_${variable}_model_${variable}_${agg_label}_${modulator}_${phase}.flag
while [ ! -f ${flagfile} ]
    do
    echo "INFO: ${jobname} is still running ! Waiting for the flag file to be written at ${flagfile}..."
    echo "INFO: check again in ${checktime} seconds..."
    sleep ${checktime}
done
echo "INFO: The flag for the job ${jobname} written by get_skill_season.py has been found !!"
echo "Exiting now..."
sleep 1
exit 0
