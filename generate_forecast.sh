#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Description: `basename $0` This script transforms raw ECMWF SEA5.1 forecasts into the percentile formate needed for visulization."
  exit 0
fi

#source ${HOME}/.bashrc
source /oceano/gmeteo/users/swen/.bashrc
#check python version
echo "Your Python version is:"
python --version

#set path to the run directory as well as to the local and remote directories containing the netCDF files to be transferred
RUNDIR=/lustre/gmeteo/PTICLIMA/Inventory/Scripts/pyPTIclima/pySeasonal
LOGDIR=${RUNDIR}/LOG

## EXECUTE #############################################################
inityear=`date '+%Y'` #forcast init year
#initmonth=`date '+%m'` #forcast init month
initmonth=`date -d "1 month ago" +'%m'`

cd ${RUNDIR}
echo "The forecast initialized on year "${inityear}" and month "${initmonth}" will be generated..."
#python pred2tercile_operational.py ${inityear} ${initmonth} > LOG/logfile_${initdate}.log
python pred2tercile_operational.py ${inityear} ${initmonth}

echo "generate_forecast.sh has run successfully, exiting now..."
exit 0
