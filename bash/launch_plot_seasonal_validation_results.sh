#!/bin/bash

####################################################################################################
# launch_plot_seasonal_validation_results.sh:
####################################################################################################
#
# Description: This script launches <plot_seasonal_validation_results.sh> to queue, which in turn calls <plot_seasonal_validation_results.py> in order to plot the verification results and generate binary skill masks used for front-end visualization
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#load your software
source ${HOME}/.bashrc

#environmental and job variables
partition=meteo_long
exclude_node=wn056
exectime=03:00:00 # 00:45:00 for plot_maps = 'no' and medcof; 04:00:00 for plot_maps = 'yes' and Iberia and 1mon; 01:30:00 for plot_maps = 'yes' and 00:30:00 for plot_maps = 'no' and Canarias and 1mon; <plot_maps> has to be set in the config files contained in the config folder !
memory=32gb #32 gb for medcof, 64gb for Iberia
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
BASHDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/bash
PYDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal

## input variables that will be passed to the python script get_skill_season.py
agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
vers='v1q'
domain='medcof'

# input variables constructed from those defined above
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG/plot/${domain}
FLAGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/FLAG/plot/${domain}

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

cd ${BASHDIR}

#loop through the aggregation windows
for agg_label in "${agg_label_list[@]}"
do
    #construct the command to be sent to queue
    QSUB="sbatch \
        --partition=${partition}
        --exclude=${exclude_node}
        --time=${exectime} \
        --job-name=plot_${agg_label}_${vers} \
        --export=ALL \
        --begin=now \
        --output=${LOGDIR}/plot_${agg_label}_${vers}.out \
        --ntasks-per-node=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=${memory} \
        ./plot_seasonal_validation_results.sh ${agg_label} ${vers} ${domain} ${RUNDIR} ${PYDIR} ${LOGDIR}" #get_skill_season.sh contains the Python script to be run on the working node
    echo ${QSUB} #prints the command sent to queue
    ${QSUB} #sent to queue  !
    sleep 5
done

echo "launch_plot_seasonal_validation_results.sh has been sent to queue successfully, exiting now..."
exit 0
