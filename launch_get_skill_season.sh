#!/bin/bash

####################################################################################################
# launch_get_skill_season.sh:
####################################################################################################
#
# Description: This script launches the model verification against observations and stores the results in netCDF format
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#send to queue with e.g.:
#sbatch --job-name=get_skill_pvpot_enso1 --partition=meteo_long --mem=32G --time=02:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 launchme.sh

#load your software
source ${HOME}/.bashrc

#environmental and job variables
partition=meteo_long
exectime=08:00:00
memory=24gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG

# #input variables that will be passed to the python script get_skill_season.py
# vers='v1m' #string format
# model_list=('ecmwf51' 'cmcc35') #bash array containing the model names and versions thereof
# agg_label_list=('1mon' '3mon') #bash array containing the temporal aggregation windows to be considered
# modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof

vers='v1m_test' #string format
model_list=('eccc5') #bash array containing the model names and versions thereof
agg_label_list=('1mon' '3mon') #bash array containing the temporal aggregation windows to be considered
modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof


# EXECUTE #######################################################################################
#check python version
echo "Your Python version is:"
python --versionv

cd ${RUNDIR}

for model in "${model_list[@]}"
do
    for agg_label in "${agg_label_list[@]}"
    do
        for modulator_plus_phase in "${modulator_plus_phase_list[@]}"
        do
            #go the the run directory and launch get_skill_season.py
            cd ${RUNDIR}
            if [[ ${modulator_plus_phase} == "enso0" ]]; then
                modulator="enso"
                phase="0"
            elif [[ ${modulator_plus_phase} == "enso1" ]]; then
                modulator="enso"
                phase="1"
            elif [[ ${modulator_plus_phase} == "enso2" ]]; then
                modulator="enso"
                phase="2"
            elif [[ ${modulator_plus_phase} == "none" ]]; then
                modulator='none'
                phase='none'
            else
                echo 'ERROR: unknown entry for <modulator_plus_phase> !'
            fi

            #construct the command to be sent to queue
            QSUB="sbatch \
                --partition=${partition}
                --time=${exectime} \
                --job-name=${vers}_${model}_${agg_label}_${modulator}_${phase} \
                --export=ALL \
                --begin=now \
                --output=${LOGDIR}/${vers}_${model}_${agg_label}_${modulator}_${phase}.out \
                --ntasks-per-node=1 \
                --ntasks=1 \
                --cpus-per-task=1 \
                --mem=${memory} \
                ./get_skill_season.sh ${vers} ${model} ${agg_label} ${modulator} ${phase} ${RUNDIR} ${LOGDIR}" #get_skill_season.sh contains the Python script to be run on the working node
            echo ${QSUB} #prints the command sent to queue
            ${QSUB} #sent to queue  !

            sleep 5
        done
    done
done

echo "launch_get_skill_season.sh has been sent to queue successfully, exiting now..."
exit 0
