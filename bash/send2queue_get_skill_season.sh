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
domain=${7}
variable=${8}
agg_label=${9}
modulator=${10}
phase=${11}
RUNDIR=${12}
BASHDIR=${13}
PYDIR=${14}
LOGDIR=${15}
FLAGDIR=${16}
jobname=${17}

## EXECUTE #########################################################################
checkflag=yes
checktime=120 #time in seconds that passed from one flag check to another

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}

echo "Passing ${jobname} to get_skill_season.sh ..."

#construct the command to be sent to queue
QSUB="sbatch \
    --partition=${partition} \
    --time=${exectime} \
    --job-name=${jobname} \
    --export=ALL \
    --begin=now \
    --output=${LOGDIR}/${jobname}.out \
    --ntasks-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=${memory} \
    --mail-user=swen.brands@gmail.com \
    --mail-type=FAIL,TIME_LIMIT \
    ./get_skill_season.sh ${vers} ${model} ${domain} ${variable} ${agg_label} ${modulator} ${phase} ${RUNDIR} ${PYDIR} ${LOGDIR} ${FLAGDIR} ${jobname}" #get_skill_season.sh contains the Python script to be run on the working node

#go to bash directory and launch the command to queue
cd ${BASHDIR}
echo ${QSUB} #prints the command sent to queue
${QSUB} #sent to queue !

#--------------------------------------------------------------------------------------------------
# Check whether get_skill_season.py executes correctly and writes the flag
#--------------------------------------------------------------------------------------------------

if [[ "${checkflag}" == "yes" ]]; then
    flagfile=${FLAGDIR}/get_skill_season_${vers}_${domain}_${model}_model_${variable}_${agg_label}_${modulator}_${phase}.flag
    while [ ! -f ${flagfile} ]
        do
        echo "INFO: ${jobname} is still running ! Waiting for the flag file to be written at ${flagfile}..."
        echo "INFO: check again in ${checktime} seconds..."
        sleep ${checktime}
    done
    echo "INFO: The flag for the job ${jobname} written by get_skill_season.py has been found !!"
elif [[ "${checkflag}" == "no" ]]; then
    echo "INFO: the <checkflag> parameter in send2queue_get_skill_season.sh is set to ${checkflag}. In this option, it does not monitor the jobs sent to queue !"
else
    echo "WARNING: the option <checkflag=${checkflag}> set in send2queue_get_skill_season.sh is not known !" >&2
fi

echo "Exiting now..."
sleep 1
exit 0
