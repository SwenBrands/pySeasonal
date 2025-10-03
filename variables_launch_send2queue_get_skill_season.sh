#!/bin/bash -l

###############################################################################################
#this script defines the variables for launch_send2queue_get_skill_season.py on the SMG cluster
#author: Swen Brands (IFCA, CSIC-UC)
###############################################################################################

#environmental and job variables
partition=meteo_long
exclude_node=wn055,wn056
exectime=00:45:00
memory=20gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
LOGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/LOG/get_skill
FLAGDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal/FLAG/get_skill

# # input variables that will be passed to the python script get_skill_season.py
# vers='v1o' #string format
# model_list=('ecmwf51' 'cmcc35') #bash array containing the model names and versions thereof
# agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
# modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof
# variable_list=('pvpot' 'fwi' 'SPEI-3-M' 't2m' 'tp' 'msl' 'si10' 'ssrd') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py

vers='v1o' #string format
model_list=('eccc5') #bash array containing the model names and versions thereof
agg_label_list=('1mon' '2mon' '3mon' '4mon' '5mon') #bash array containing the temporal aggregation windows to be considered
modulator_plus_phase_list=('none' 'enso0' 'enso1' 'enso2') #bash array containing all modulators and phases thereof
variable_list=('pvpot' 'fwi' 't2m' 'tp' 'msl' 'si10' 'ssrd') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py
