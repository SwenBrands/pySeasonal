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
exectime=00:45:00 # 09:00:00 for plot_maps = 'yes' and 1mon
memory=16gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
BASHDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/bash
PYDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG/plot
FLAGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/FLAG/plot


# input variables that will be passed to the python script get_skill_season.py
agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
plot_maps='no'
vers='v1p'

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
        ./plot_seasonal_validation_results.sh ${agg_label} ${plot_maps} ${vers} ${RUNDIR} ${PYDIR} ${LOGDIR}" #get_skill_season.sh contains the Python script to be run on the working node
    echo ${QSUB} #prints the command sent to queue
    ${QSUB} #sent to queue  !
    sleep 5
done

echo "launch_plot_seasonal_validation_results.sh has been sent to queue successfully, exiting now..."
exit 0
