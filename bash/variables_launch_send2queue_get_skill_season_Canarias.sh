#!/bin/bash -l

###############################################################################################
#this script defines the variables for launch_send2queue_get_skill_season.py on the SMG cluster
#author: Swen Brands (IFCA, CSIC-UC)
###############################################################################################

# domain identifier
domain='Canarias' #character string specifying the spatial domain for which the verification will be applied

#environmental and job variables
partition=meteo_long
exclude_node=wn055
exectime=00:35:00 #00:12:00 for agg_label_list=('1mon') and modulator_plus_phase_list=('none') # 00:30:00 for agg_label_list=('1mon') and modulator_plus_phase_list=('none')
memory=24gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/pyseasonal
BASHDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/bash
PYDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/pyseasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG/get_skill/${domain}
FLAGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/FLAG/get_skill/${domain}

# input variables that will be passed to the python script get_skill_season.py
vers='v1r' #string format
model_list=('ecmwf51') #bash array containing the model names and versions thereof
# agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
agg_label_list=('1mon') #bash array containing the temporal aggregation windows to be considered
# modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof
modulator_plus_phase_list=('none') #bash array containing all modulators and phases thereof
# variable_list=('Rx1day-C4' 'Rx5day-C4' 'TNm-C4' 'PRtot-C4' 'PRm-C4' 'TXm-C4' 'FD-C4' 'SU-C4' 'TR-C4') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py
variable_list=('TNm-C4') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py
