#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Description: `basename $0` This script calls the Python script <pred2tercile_operational.py>, which transforms raw daily seasonal forecasts form CDS into percentile format needed for visulization within the PTI-Climate project."
  exit 0
fi

#source ${HOME}/.bashrc
source /oceano/gmeteo/users/swen/.bashrc
#check python version
echo "Your Python version is:"
python --version

initdate_list=('202401' '202402' '202403' '202404' '202405' '202406' '202407' '202408' '202409' '202410')

# inityear=`date '+%Y'` #forcast init year
# initmonth=`date '+%m'` #forcast init month

#set path to the run directory as well as to the local and remote directories containing the netCDF files to be transferred
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=${RUNDIR}/LOG

## EXECUTE #############################################################
cd ${RUNDIR}
for initdate in "${initdate_list[@]}"
do
  echo "The forecast initialized on "${initdate}" will be generated..."
  python pred2tercile_operational.py ${initdate}
done

echo "launch_pred2tercile_operational.sh has run successfully, exiting now..."
exit 0
