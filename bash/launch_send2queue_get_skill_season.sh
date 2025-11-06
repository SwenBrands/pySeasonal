#!/bin/bash

####################################################################################################
# launch_send2queue_get_skill_season.sh:
####################################################################################################
#
# Description: This script launches the get_skill_season workflow: For each model, temporal aggregation and climate modulator+phase,
# a separate job is sent to queue by send2queue_get_skill_season.sh. The latter script calls get_skill_season.sh on the working node,
# which in turn executes get_skill_season.py there. send2queue_get_skill_season.sh sends a procsse to the frontal nodes that checks
# whether get_skill_season.py has completed successfully, in which case it writes a flag. Once found, the process on the frontal node closes.
# CONCLUSION: An open process running a long time on the frontal node and writing into the logfile indicates that the correpsonding job sent to queue did not finish successfully.
#
# execute with:
#
# ./launch_send2queue_get_skill_season.sh variables_launch_send2queue_get_skill_season.sh > get_skill_season_workflow.log 2>&1
#
# Author: Swen Brands (IFCA, CSIC-UC)
####################################################################################################

#--------------------------------------------------------------------------------------------------
# Check if the script is called correctly
#--------------------------------------------------------------------------------------------------
#load input variables from variables_launch_send2queue_get_skill_season.sh
FILE_VARIABLES=${1} #read the variables file

JOB_NAME="$(basename "$0")"
MESSAGE="Usage: ./${JOB_NAME} [variables file]"

if [ ! -f "${FILE_VARIABLES}" ]
then 
    echo "I cannot find the variables file ${FILE_VARIABLES}"
    echo ${MESSAGE}
    exit 1
fi

#load your software
source ${HOME}/.bashrc
source ${FILE_VARIABLES} #load the variables into memory
ulimit -u 1024

#print the just loaded variables
echo "--------------------------------------------------------------------------------"
echo "launch_sent2queue_get_skill_season.sh will be run with the following variables:"
echo "--------------------------------------------------------------------------------"
echo "partition: "${partition}
echo "exclude_node: "${exclude_node}
echo "exectime: "${exectime}
echo "memory: "${memory}
echo "RUNDIR: "${RUNDIR}
echo "BASHDIR: "${BASHDIR}
echo "PYDIR: "${PYDIR}
echo "LOGDIR: "${LOGDIR}
echo "FLAGDIR: "${FLAGDIR}
echo "vers: "${vers}
echo "model_list: "${model_list[@]}
echo "domain_for_config: "${domain_for_config}
echo "agg_label_list: "${agg_label_list[@]}
echo "modulator_plus_phase: "${modulator_plus_phase_list[@]}
echo "variable_list: "${variable_list[@]}
echo "--------------------------------------------------------------------------------"
echo "--------------------------------------------------------------------------------"

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
                #define the job name
                jobname=${vers}_${model}_${variable}_${agg_label}_${modulator}_${phase}
                echo "Passing ${jobname} to send2queue_get_skill_season.sh ..."              
                #send2queue_get_skill_seasons will send the model evalatuion to queue and is itself sent into the background of the frontal node
                . ${BASHDIR}/send2queue_get_skill_season.sh ${partition} ${exclude_node} ${exectime} ${memory} ${vers} ${model} ${domain_for_config} ${variable} ${agg_label} ${modulator} ${phase} ${RUNDIR} ${BASHDIR} ${PYDIR} ${LOGDIR} ${FLAGDIR} ${jobname} > ${LOGDIR}/send2queue_get_skill_season_${jobname}.log 2>&1 &
                sleep 60
            done
        done
    done
done

echo "launch_send2queue_get_skill_season.sh has been sent to queue successfully, exiting now..."
exit 0
