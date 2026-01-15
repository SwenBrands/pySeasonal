#!/bin/bash -l

###############################################################################################
#this script defines the variables for launch_send2queue_get_skill_season.py on the SMG cluster
#author: Swen Brands (IFCA, CSIC-UC)
###############################################################################################

# domain identifier
domain='Iberia' #character string specifying the spatial domain for which the verification will be applied

#environmental and job variables
partition=meteo_long
exclude_node=wn055
exectime=01:55:00 #05:00:00 for agg_label_list=('1mon') and modulator_plus_phase_list=('none'); #04:00:00 for agg_label_list=('2mon') and modulator_plus_phase_list=('none')
memory=48gb #agg_label_list=('2mon') and modulator_plus_phase_list=('none') successfully tested with 128gb; 64gb for agg_label_list=('1mon') and modulator_plus_phase_list=('enso0'), for agg_label_list=('1mon') and modulator_plus_phase_list=('none') tested sucessfully with 144 or 156gb
RUNDIR=/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal
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
# variable_list=('Rx5day-C4_up010' 'TNm-C4_up010' 'PRtot-C4_up010' 'PRm-C4_up010' 'TXm-C4_up010' 'FD-C4_up010' 'SU-C4_up010' 'TR-C4_up010') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py
variable_list=('SPEI-3-M-C4_up010') #bash array of variables to be processed; must coincide with <variables_gcm> in aggregate_hindcast.py
