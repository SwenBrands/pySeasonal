#!/bin/bash


####################################################################################################
# launch_get_skill_season.sh:
####################################################################################################
#
# Description: This script is launched to queue with launch_plot_seasonal_validation_results.sh. Itsefl, it launches the Python script plot_seasonal_validation_results.py, which plots the verification results obtained with get_skill_season.py and also generates binary skill masks in netCDF format.
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

#include helf option
if [ "$1" == "-h" ]; then
  echo "Description: `basename $0` This script is launched to queue with launch_plot_seasonal_validation_results.sh. It launches the Python script plot_seasonal_validation_results.py, which plots the verification results obtained with get_skill_season.py and also generates binary skill masks in netCDF format."
  exit 0
fi

#load your envirmonent
source ${HOME}/.bashrc

#input variables that will be passed to the python script get_skill_season.py
agg_label=${1}
plot_maps=${2}
vers=${3}
RUNDIR=${4}
PYDIR=${5}
LOGDIR=${6}

## EXECUTE #########################################################################
log_label=plot_seasonal_validation_results_${agg_label}
#check python version
echo "Your Python version is:"
python --version

#go to the run directory and launch get_skill_season.py
cd ${RUNDIR}
# current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# run the Python script
python ${PYDIR}/plot_seasonal_validation_results.py ${agg_label} ${plot_maps} ${vers} > ${LOGDIR}/${log_label}.log

echo "plot_seasonal_validation_results.sh has been sent to queue successfully, exiting now..."
sleep 1
exit 0
