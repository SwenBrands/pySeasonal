#!/usr/bin/env python
''' This scipts loads the tercile probabilities for a single model and forecast previously obtained with pred2tercile_operational.py, adds a single skill mask with the same dimensions to the array and saves a new file in netCDF format.'''

#load packages
from datetime import date
import numpy as np
import xarray as xr
import os
import pandas as pd
import dask
import sys
import pdb
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#the init of the forecast (year and month) can passed by bash; if nothing is passed these parameters will be set by python
if len(sys.argv) > 1:
    print("Reading from input parameters passed via bash")
    year_init = sys.argv[1]
    month_init = sys.argv[2]
    if len(year_init) != 4 or len(month_init) != 2:
        raise Exception('ERROR: check length of <year_month_init> input parameter !')
else:
    print("No input parameter have been provided by the user and the script will set the <year_init> and <month_init> variables for the year and month of the current date...")
    year_init = str(date.today().year)
    month_init = f"{date.today().month:02d}"
    print(date.today())
print(year_init, month_init)

year_init = 2024 #a list containing the years the forecast are initialized on, will be looped through with yy
month_init = 10 #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

#set input parameters
val_vers = 'v1l_mon' #version number of the validation results
model = ['ecmwf'] # ['cmcc','ecmwf'] list containing the acronyms of the models to be assessed
version = ['51'] # ['35','51'] list containgin the versions of these models
obs = 'era5' #string containing the observational reference
years_quantile = [[1993,2022],[1993,2022]] #years used to calculate the quantiles with get_skill_season.py; list containing as many sublists as they are models
years_validation = [1981,2022] #years the verification has been done for using get_skill_season.py
season_length = 1 #length of the season in months, e.g. 3 for DJF, JFM, etc.
subperiod = 'none' # climate oscillation assumed to modulate the verification results; currently: "enso" or "none"
score = 'pearson_r_binary' #score of the binary skill mask that will be stored by this script

# variable_qn = ['pvpot','SPEI-3-M','fwi','msl','t2m','tp','si10','ssrd'] # variable name used inside and outside of the quantile file. This is my work and is thus homegeneous.
# variable_fc = ['pvpot','SPEI-3-M','fwi','psl','tas','pr','sfcWind','rsds'] # variable name used in the file name, i.e. outside the file, ask collegues for data format harmonization
# variable_fc_nc = ['pvpot','SPEI-3-M','FWI','psl','tas','pr','sfcWind','rsds'] # variable name within the model netcdf file, may vary depending on source
# time_name = ['time','time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'] #name of the time dimension within the model netcdf file, may vary depending on source
# lon_name = ['lon','lon','lon','x','x','x','x','x']
# lat_name = ['lat','lat','lat','y','y','y','y','y']
# file_start = ['seasonal-original-single-levels','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'] #start string of the file names

variables = ['SPEI-3-M','pr','tas'] # variable name used inside and outside the forecast file, ask collegues for data format harmonization
# time_name = ['timer'] #name of the dimension indicating the init time in the forecast file, <timer> refers to "reference time"
# lon_name = ['x']
# lat_name = ['y']

datatype = 'float32' #data type of the variables in the output netcdf files
domain = 'iberia' #spatial domain
detrended = 'no' #yes or no, linear detrending of the gcm and obs time series prior to validation
corr_outlier = 'no'
nr_mem = [25] #considered ensemble members, not yet in use !

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

#set basic path structure for observations and gcms
gcm_store = 'lustre' #argo, laptop, F, extdisk2 or lustre

## EXECUTE #############################################################
quantile_threshold = [0.33,0.67]
# #check consistency of input parameters
# if (lon_name[-1] != 'x') | (lat_name[-1] != 'y'):
    # raise Exception('ERROR: the last entries of the <lon_name> and <lat_name> input lists must be "x" and "y" respectively to ensure that the output netCDF file is consistent!')

#set path to input gcm files
if gcm_store == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    path_gcm_base = home+'/DATA/SEASONAL/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = home+'/DATA/SEASONAL/seasonal-original-single-levels_derived' # head directory of the source files
    path_gcm_base_masked = home+'/DATA/SEASONAL/seasonal-original-single-levels_masked' # head directory of the source files    
    rundir = home+'/Scripts/SBrands/pyPTIclima/pySeasonal'
    dir_validation = home+'/Results/seasonal/validation/'+val_vers
    dir_forecast = home+'/Results/seasonal/forecast'
    dir_output = dir_forecast
elif gcm_store == 'argo':
    home = os.getenv("DATA_DIR", "")
    path_gcm_base = home+'seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = home+'seasonal-original-single-levels_derived' # head directory of the source files
    path_gcm_base_masked = home+'seasonal-original-single-levels_masked' # head directory of the source files
    dir_validation = '/tmp/terciles/'
    dir_forecast = home+'seasonal-original-single-levels_derived/medcof/forecast/terciles/'
    dir_output = dir_forecast
else:
    raise Exception('ERROR: unknown entry for <path_gcm_base> !')
print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

#go to rundir
os.chdir(rundir)

#create output directory of the forecasts generated here, if it does not exist.
if os.path.isdir(dir_forecast) != True:
    os.makedirs(dir_forecast)

#load the skill mask and filter out target score to be added to the output array
filename_validation = dir_validation+'/binary_validation_results_pticlima_'+domain+'_'+str(years_validation[0])+'_'+str(years_validation[-1])+'_'+val_vers+'.nc'
nc_val = xr.open_dataset(filename_validation)

#make forecast for each model and variable

#fc_np_arr = np.zeros(#numpy array containing the probability forecasts

for mm in np.arange(len(model)):
    #load the forecast
    filename_forecast = dir_forecast+'/probability_'+model[mm]+version[mm]+'_init_'+str(year_init)+str(month_init).zfill(2)+'_'+str(season_length)+'mon_dtr_'+detrended+'_refyears_'+str(years_quantile[mm][0])+'_'+str(years_quantile[mm][1])+'.nc'
    nc_forecast = xr.open_dataset(filename_forecast)

    #check whether the previously stored model and version thereof match those requested by this script
    if nc_forecast.model == model[mm]+version[mm]:
        print('The requested model '+model[mm]+' and its version '+version[mm]+' coincide with the entry previously stored in '+filename_forecast+' !')
    else:
        raise ValueError('The requested model '+model[mm]+' and its version '+version[mm]+' do NOT coincide with the entry previously stored in '+filename_forecast+' !')
    
    #filter subperiod, detrending option, model and version, season, and variable from skill mask
    fc_season_length = nc_forecast.season.length_in_months
    fc_seasons = nc_forecast.season.values
    nc_val_sub = nc_val.sel(subperiod=subperiod,detrended=detrended,model=model[mm]+version[mm],season=fc_seasons,variable=variables)[score]
    val_leads = nc_val_sub.lead.values

    # #check whether the number of seasons provided by the forrecast files matches the number of lead-times provided by the skill file 
    # if len(fc_seasons) != len(val_versleads):
    #     raise ValueError('the <seasons> and <leads> variables must have the same lenght !')    
    
    skill_mask = np.zeros((len(variables),len(fc_seasons),len(nc_val.y),len(nc_val.x)))
    for sea in np.arange(len(fc_seasons)):
        skill_mask_step = nc_val_sub.sel(season=fc_seasons[sea],lead=val_leads[sea])
        skill_mask[:,sea,:,:] = skill_mask_step.values

    #bring numpy array into xarray data array format
    skill_mask = xr.DataArray(skill_mask.astype('float32'),coords=[variables,fc_seasons,nc_val.y,nc_val.x],dims=['variable','season','y','x'], name='skill_mask')

    #cut down the forecast domain to the skill domain indicated in the <domain> input variable
    nc_forecast = nc_forecast.sel(y=skill_mask.y,x=skill_mask.x)
    
    #merge probabilies and skill mask
    nc_forecast['skill_mask'] = skill_mask
    
    #add attributes
    nc_forecast.skill_mask.attrs['long_name'] = 'skill_based_on_'+score
    
    # save the mergerd output file
    savename = dir_forecast+'/probability_plus_skill_'+model[mm]+version[mm]+'_init_'+str(year_init)+str(month_init).zfill(2)+'_'+str(fc_season_length)+'mon_dtr_'+detrended+'_refyears_'+str(nc_forecast.tercile.tercile_period[0])+'_'+str(nc_forecast.tercile.tercile_period[1])+'.nc'
    encoding = dict(probability=dict(chunksizes=(1, 1, 1, 1, len(nc_forecast.y), len(nc_forecast.y)))) #https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
    nc_forecast.to_netcdf(savename,encoding=encoding)

    #clean up
    nc_val_sub.close()
    skill_mask.close()
    nc_forecast.close()
    #del(nc_val_sub,skill_mask,nc_forecast)

#clean up
nc_val.close()
#del(nc_val)

print('INFO: assign_skill2forecast.py has been run successfully ! The netcdf output file has been stored at '+dir_forecast)
quit()
