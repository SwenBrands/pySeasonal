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
exclude_node=wn56
exectime=00:45:00
memory=20gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG
FLAGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/FLAG

# input variables that will be passed to the python script get_skill_season.py
vers='v1n' #string format
model_list=('ecmwf51' 'cmcc35') #bash array containing the model names and versions thereof
agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof
variable_list=('pvpot' 'fwi' 'SPEI-3-M' 't2m' 'tp' 'msl' 'si10' 'ssrd') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py

# vers='v1m_test' #string format
# model_list=('ecmwf51') #bash array containing the model names and versions thereof
# agg_label_list=('5mon') #bash array containing the temporal aggregation windows to be considered
# modulator_plus_phase_list=('none') #bash array containing all modulators and phases thereof
# variable_list=('pvpot') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py

# EXECUTE #######################################################################################
#check python version
echo "Your Python version is:"
python --version

#clean the LOG and FLAG directories
rm -r ${LOGDIR}
rm -r ${FLAGDIR}
mkdir ${LOGDIR}
mkdir ${FLAGDIR}
sleep 1

cd ${RUNDIR}

#loop through the models
for model in "${model_list[@]}"
do
    #loop through the aggregation windows
    for agg_label in "${agg_label_list[@]}"
    do
        #loop through the modulators and their phases
        for modulator_plus_phase in "${modulator_plus_phase_list[@]}"
        do                        
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
            #loop through the variables
            for variable in "${variable_list[@]}"
            do
                #construct the command to be sent to queue
                QSUB="sbatch \
                    --partition=${partition}
                    --exclude=${exclude_node}
                    --time=${exectime} \
                    --job-name=${vers}_${model}_${variable}_${agg_label}_${modulator}_${phase} \
                    --export=ALL \
                    --begin=now \
                    --output=${LOGDIR}/${vers}_${model}_${variable}_${agg_label}_${modulator}_${phase}.out \
                    --ntasks-per-node=1 \
                    --ntasks=1 \
                    --cpus-per-task=1 \
                    --mem=${memory} \
                    ./get_skill_season.sh ${vers} ${model} ${variable} ${agg_label} ${modulator} ${phase} ${RUNDIR} ${LOGDIR}" #get_skill_season.sh contains the Python script to be run on the working node
                echo ${QSUB} #prints the command sent to queue
                ${QSUB} #sent to queue  !
                sleep 10
            done
        done
    done
done

echo "launch_get_skill_season.sh has been sent to queue successfully, exiting now..."
exit 0
