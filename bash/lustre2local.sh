#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

####################################################################################################
# lustre2local.sh:
####################################################################################################
#
# Description: Downloads all files necessary to run pred2tercile_operational.py on your local system
# from the Lustre file system of the SMG UI cluster 
#
# Author: Swen Brands (CSIC-UC)
####################################################################################################

# load your environment
source ${HOME}/.bashrc

#get current year and month in YYYYMM format
YYYYMM=$(date +"%Y%m")
#YYYYMM=202510

# input variables that will be passed to the python script get_skill_season.py
RUNDIR="${HOME}/datos/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal"
STORE_LOCAL="${HOME}/datos/PTICLIMA/DATA/SEASONAL"
STORE_REMOTE="/lustre/gmeteo/PTICLIMA/DATA/SEASONAL"
STORE_QUAN_REMOTE="/lustre/gmeteo/PTICLIMA/Results/seasonal/validation/v1p"
STORE_QUAN_LOCAL="${HOME}/datos/PTICLIMA/Results/seasonal/validation/v1p"
STORE_MASK_REMOTE="/lustre/gmeteo/PTICLIMA/Auxiliary-material/Masks"
STORE_MASK_LOCAL="${HOME}/datos/PTICLIMA/Auxiliary-material"
STORE_FC_REMOTE="/lustre/gmeteo/PTICLIMA/Results/seasonal/forecast" # path to the remote forecast files
STORE_FC_LOCAL="${HOME}/datos/PTICLIMA/Results/seasonal/forecast"

agg_list=('1mon' '2mon' '3mon' '4mon' '5mon')
# model_list=('eccc' 'ecmwf' 'cmcc') #bash array containing the model names and versions thereof
model_list=('ecmwf') #bash array containing the model names and versions thereof
variable_list=('pvpot' 'fwi' 'SPEI-3-M' 'tas' 'pr' 'psl' 'sfcWind' 'rsds')
# variable_list=('tas' 'pr' 'psl' 'sfcWind' 'rsds')
variable_single=('tas' 'pr' 'psl' 'sfcWind' 'rsds') #standard variables
variable_derived=('pvpot' 'fwi') #derived variables
variable_masked=('SPEI-3-M') #masked variables
domain='medcof'

cd "${RUNDIR}"

#create sub-directories, first loop
for model in "${model_list[@]}"
do
    if [[ "${model}" == "ecmwf" ]]; then
        version="51"
    elif [[ "${model}" == "cmcc" ]]; then
        version="4"
    elif [[ "${model}" == "eccc" ]]; then
        version="5"
    else
        echo "Unknown entry value for ${model}, exiting now !"
        exit 1
    fi
    #second loop (nested)
    for variable in "${variable_list[@]}"
    do
        if [[ " ${variable_single[@]} " =~ " ${variable} " ]]; then
            extension=seasonal-original-single-levels/${domain}/forecast/${variable}/${model}/${version}
        elif [[ " ${variable_derived[@]} " =~ " ${variable} " ]]; then
            extension=seasonal-original-single-levels_derived/${domain}/forecast/${variable}/${model}/${version}
        elif [[ " ${variable_masked[@]} " =~ " ${variable} " ]]; then
            extension=seasonal-original-single-levels_masked/${domain}/forecast/${variable}/${model}/${version}/coefs_pool_members
        else
            echo "Unknown entry value for ${variable}, exiting now !"
        fi
        remotedir="${STORE_REMOTE}/${extension}/${YYYYMM}"
        tardir="${STORE_LOCAL}/${extension}"

        #create target directory, go there and download target file from lustre
        mkdir -p "${tardir}"
        echo "downloading ${remotedir} -> ${tardir}"
        rsync -av --timeout=7 "swen@ui.sci.unican.es:${remotedir}" "${tardir}"
        # rsync -avz --ignore-missing-args swen@ui.sci.unican.es:${remotedir} ${tardir}
        # rsync -avz --dry-run "swen@ui.sci.unican.es:${remotedir} ${tardir}"
        sleep 8
    done
done

#download the quantiles
for agg in "${agg_list[@]}"
do
    echo "aggregation is ${agg}"
    extension_quan="${agg}/quantiles"
    remotedir_quan="${STORE_QUAN_REMOTE}/${extension_quan}"
    tardir_quan="${STORE_QUAN_LOCAL}/${agg}"
    mkdir -p "${tardir_quan}"
    echo "downloading ${remotedir_quan} -> ${tardir_quan}"
    rsync -av --timeout=7 "swen@ui.sci.unican.es:${remotedir_quan}" "${tardir_quan}"
    sleep 8
done

# download the land-sea mask
mkdir -p "${STORE_MASK_LOCAL}"
rsync -av --timeout=7 "swen@ui.sci.unican.es:${STORE_MASK_REMOTE}" "${STORE_MASK_LOCAL}"

# create the local output directory where the forecasts are stored
mkdir -p "${STORE_FC_LOCAL}"

exit 0
