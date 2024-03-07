#!/usr/bin/env python
''' This script loads a forecast at a given init, aggregates the raw values to seasonal averages and transforms them into terciles.'''

#load packages
import numpy as np
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import os
import xesmf
import pandas as pd
import dask
from scipy.signal import detrend
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters
quantile_version = '1i' #version number of the quantiles file used here
model = 'ecmwf' #interval between meridians and parallels
version = '51'

#year_init = [2023,2023,2023,2023,2023,2023,2023,2023,2023,2023,2023,2023,2024,2024] #a list containing the years the forecast are initialized on, will be looped through with yy
#month_init = [1,2,3,4,5,6,7,8,9,10,11,12,1,2] #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

year_init = [2024] #a list containing the years the forecast are initialized on, will be looped through with yy
month_init = [2] #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

years_quantile = [1981,2022] #years used to calculate the quantiles with get_skill_season.py
season_length = 3 #length of the season, e.g. 3 for DJF, JFM, etc.

variable_qn = ['SPEI-3','fwi','msl','t2m','tp','si10','ssrd'] # variable name used inside and outside of the quantile file. This is my work and is thus homegeneous.
variable_fc = ['SPEI-3','fwi','psl','tas','pr','sfcWind','rsds'] # variable name used in the file name, i.e. outside the file, ask collegues for data format harmonization
variable_fc_nc = ['SPEI-3','FWI','psl','tas','pr','sfcWind','rsds'] # variable name within the model netcdf file, may vary depending on source
time_name = ['time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'] #name of the time dimension within the model netcdf file, may vary depending on source
lon_name = ['lon','lon','x','x','x','x','x']
lat_name = ['lat','lat','y','y','y','y','y']
file_start = ['seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'] #start string of the file names

precip_threshold = 1/90 #seasonal mean daily precipitation threshold in mm below which the modelled and quasi-observed monthly precipitation amount is set to 0. Bring this in exact agreement with get_skill_season.py in future versions
datatype = 'float32' #data type of the variables in the output netcdf files
domain = 'medcof' #spatial domain
detrended = 'no' #yes or no, linear detrending of the gcm and obs time series prior to validation
nr_mem = [25] #considered ensemble members, not yet in use !

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

#set basic path structure for observations and gcms
gcm_store = 'lustre' #laptop, F, extdisk2 or lustre
product = 'forecast'

## EXECUTE #############################################################
quantile_threshold = [0.33,0.67]
# #check consistency of input parameters
# if (lon_name[-1] != 'x') | (lat_name[-1] != 'y'):
    # raise Exception('ERROR: the last entries of the <lon_name> and <lat_name> input lists must be "x" and "y" respectively to ensure that the output netCDF file is consistent!')

#set path to input gcm files
if gcm_store == 'laptop':
    home = os.getenv('HOME')
    path_gcm_base = home+'/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    rundir = home+'/lustre/gmeteo/PTICLIMA/Inventory/Scripts/pyPTIclima/pySeasonal'
    dir_quantile = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation'
    dir_forecast = home+'/datos/tareas/proyectos/pticlima/seasonal/results/forecast'
elif gcm_store == 'F':
    home = os.getenv('HOME')
    path_gcm_base = '/media/swen/F/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    rundir = home+'/lustre/gmeteo/PTICLIMA/Inventory/Scripts/pyPTIclima/pySeasonal'
    dir_quantile = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation'
    dir_forecast = home+'/datos/tareas/proyectos/pticlima/seasonal/results/forecast'
elif gcm_store == 'extdisk2':
    home = os.getenv('HOME')
    path_gcm_base = '/media/swen/ext_disk2/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    path_gcm_base = '/media/swen/ext_disk2/datos/GCMData/seasonal-original-single-levels/'+domain # head directory of the source files
    rundir = home+'/lustre/gmeteo/PTICLIMA/Inventory/Scripts/pyPTIclima/pySeasonal'
    dir_quantile = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation'
    dir_forecast = home+'/datos/tareas/proyectos/pticlima/seasonal/results/forecast'
elif gcm_store == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    path_gcm_base = home+'/DATA/SEASONAL/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = home+'/DATA/SEASONAL/seasonal-original-single-levels_derived' # head directory of the source files
    path_gcm_base_masked = home+'/DATA/SEASONAL/seasonal-original-single-levels_masked' # head directory of the source files    
    rundir = home+'/Inventory/Scripts/pyPTIclima/pySeasonal'
    dir_quantile = home+'/Inventory/Results/seasonal/validation'
    dir_forecast = home+'/Inventory/Results/seasonal/forecast'
else:
    raise Exception('ERROR: unknown entry for <path_gcm_base> !')
print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

#create output directory of the forecasts generated here, if it does not exist.
if os.path.isdir(dir_forecast) != True:
    os.makedirs(dir_forecast)

#load the quantiles file
filename_quantiles = dir_quantile+'/quantiles_pticlima_'+domain+'_'+str(years_quantile[0])+'_'+str(years_quantile[1])+'_v'+quantile_version+'.nc'
nc_quantile = xr.open_dataset(filename_quantiles)

for yy in np.arange(len(year_init)):
    for vv in np.arange(len(variable_fc)):
        
        #load forecast file
        if variable_fc[vv] == 'SPEI-3':
            filename_forecast = path_gcm_base_masked+'/'+domain+'/'+product+'/'+variable_fc[vv]+'/'+model+'/'+version+'/coefs_all_members/'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'/'+file_start[vv]+'_'+domain+'_'+product+'_'+variable_fc[vv]+'_'+model+'_'+version+'_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'.nc'
        elif variable_fc[vv] == 'fwi':
            filename_forecast = path_gcm_base_derived+'/'+domain+'/'+product+'/'+variable_fc[vv]+'/'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'/'+file_start[vv]+'_'+domain+'_'+product+'_'+variable_fc[vv]+'_'+model+'_'+version+'_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'.nc'
        elif variable_fc[vv] in ('psl','sfcWind','tas','pr','rsds'):
            filename_forecast = path_gcm_base+'/'+domain+'/'+product+'/'+variable_fc[vv]+'/'+model+'/'+version+'/'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'/'+file_start[vv]+'_'+domain+'_'+product+'_'+variable_fc[vv]+'_'+model+'_'+version+'_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'.nc'
        else:
            raise Exception('ERROR: check entry for variables[vv] !')
        
        #filename_forecast = path_gcm_base+'/'+product+'/'+variable_fc[vv]+'/'+model+'/'+version+'/'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'/'+file_start[vv]+'_'+domain+'_'+product+'_'+variable_fc[vv]+'_'+model+'_'+version+'_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'.nc'
        nc_fc = xr.open_dataset(filename_forecast)
        
        #check if the latitudes are in the right order or must be flipped to be consistent with the obserations used for validation
        if nc_fc[lat_name[vv]][0].values < nc_fc[lat_name[vv]][-1].values:
            print('WARNING: the latitudes in '+filename_forecast+' come in ascending order and are inverted to be consistent with the order of the remaining datasets / variables (descending) !')
            if lat_name[vv] == 'lat':
                nc_fc = nc_fc.reindex(lat=list(reversed(nc_fc.lat)))
            elif lat_name[vv] == 'y':
                nc_fc = nc_fc.reindex(y=list(reversed(nc_fc.y)))
            else:
                raise Exception('ERROR: unexpected entry for <lat_name[vv]> !')

        #transform GCM variables and units, if necessary
        nc_fc, file_valid = transform_gcm_variable(nc_fc,variable_fc[vv],variable_qn[vv],model,version)
        #check whether there is a problem with the variable units as revealed in transform_gcm_variable()
        if file_valid == 0:
            raise Exception('ERROR: There is a problem with the expected variable units in '+filename_forecast+' !')
        
        #get forecast seasons from file
        dates_fc = pd.DatetimeIndex(nc_fc.time.values)
        #check whether the model input data is monthly, otherwise daily is assumed and this must be improved in future versions
        if len(dates_fc.month) == len(np.unique(dates_fc.month)):
            print('INFO: the model input data for '+variable_fc[vv]+' is monthly !')
            months_fc_uni = dates_fc.month
        else:
            months_fc_uni = dates_fc[15::30].month #get unique forecast month in the right order, i.e. as appears in the file
        
        season = []
        season_label = []
        season_start_month = np.arange(len(months_fc_uni)-season_length+1) #index of the first month of each 3-month period
        
        #init final output array
        if vv == 0:
            out_arr = np.zeros((len(variable_fc),len(quantile_threshold)+1,len(season_start_month),len(nc_fc[lat_name[vv]]),len(nc_fc[lon_name[vv]])),dtype=datatype)
            out_arr[:] = np.nan
        for mo in season_start_month:
            season_i = months_fc_uni[mo:mo+season_length].to_list()
            season_i_label = assign_season_label(season_i)
            season_ind = np.where(np.isin(dates_fc.month,season_i))[0]
            if time_name[vv] == 'forecast_time':
                seas_mean = nc_fc[variable_fc_nc[vv]].isel(forecast_time=season_ind).mean(dim=time_name[vv])
            elif time_name[vv] == 'time':
                seas_mean = nc_fc[variable_fc_nc[vv]].isel(time=season_ind).mean(dim=time_name[vv])
            else:
                raise Exception('ERROR: unkwnown entry for <time_name[vv]> input parameter!')
            
            #precipitation correction, is here done on seasonal timescale, but must be done on montly timescale in the future to be consistent with get_skill_season.py!
            if variable_fc[vv] == 'pr':
                print('INFO: setting seasonal mean '+variable_fc[vv]+' values from '+model+ '< '+str(precip_threshold)+' to 0...')
                #zero_mask = seas_mean[variable_fc[vv]].values < precip_threshold
                #seas_mean[variable_fc[vv]].values[zero_mask] = 0.
                zero_mask = seas_mean.values < precip_threshold
                seas_mean.values[zero_mask] = 0.

            #get the ensemble quantiles for this season and leadtime (note that the season and leadtime have the same index !)
            lower_xr = nc_quantile.sel(detrended=detrended,model=model+version,quantile_threshold=quantile_threshold[0],variable=variable_qn[vv],season=season_i_label).quantile_ensemble.isel(lead=mo) #is a 2D xarray data array
            upper_xr = nc_quantile.sel(detrended=detrended,model=model+version,quantile_threshold=quantile_threshold[1],variable=variable_qn[vv],season=season_i_label).quantile_ensemble.isel(lead=mo)
            lower_np = np.tile(lower_xr.values,(seas_mean.shape[0],1,1))
            upper_np = np.tile(upper_xr.values,(seas_mean.shape[0],1,1))
            
            # upper_ind = seas_mean.where(seas_mean > upper_np)
            # upper_ind = upper_ind.where(np.isnan(upper_ind),other=1)
            # center_ind = seas_mean.where((seas_mean >= lower_np) & (seas_mean <= upper_np))
            # center_ind = center_ind.where(np.isnan(center_ind),other=1)
            # lower_ind = seas_mean.where(seas_mean < upper_np)
            # lower_ind = lower_ind.where(np.isnan(lower_ind),other=1)
            
            # upper_ind = seas_mean > upper_np
            # center_ind = (seas_mean >= lower_np) & (seas_mean <= upper_np)
            # lower_ind = seas_mean < lower_np

            upper_ind = (seas_mean > upper_np) & (~np.isnan(upper_np))
            center_ind = (seas_mean >= lower_np) & (seas_mean <= upper_np) & (~np.isnan(upper_np))
            lower_ind = (seas_mean < lower_np) & (~np.isnan(lower_np))
            
            #sum members in each category
            nr_mem = upper_ind.shape[0]
            upper_nr = upper_ind.sum(dim='member')/nr_mem
            center_nr = center_ind.sum(dim='member')/nr_mem
            lower_nr = lower_ind.sum(dim='member')/nr_mem
            
            ##set ocean points to nan
            # upper_nr = upper_nr.where(~np.isnan(lower_xr.values))
            # center_nr = center_nr.where(~np.isnan(lower_xr.values))
            # lower_nr = lower_nr.where(~np.isnan(lower_xr.values))
            
            #stack and turn to numpy format
            terciles = np.stack((lower_nr.values,center_nr.values,upper_nr.values),axis=0)
            terciles_nan = np.zeros((terciles.shape))
            terciles_nan[:] = np.nan
            #get index of the tercile having the maximum forecast probability at each grid box; save this probability at the given tercile; at the other terciles the nan value will be kept.
            for ii in np.arange(terciles.shape[1]):
                for jj in np.arange(terciles.shape[2]):
                    maxind = np.argmax(terciles[:,ii,jj])
                    terciles_nan[maxind,ii,jj] = terciles[maxind,ii,jj]
                    terciles_nan[terciles_nan == 0] = np.nan #set 0 probabilities to nan
            #terciles_nan[maxprob_ind] = terciles[maxprob_ind,:,:]
            out_arr[vv,:,mo,:,:] = terciles_nan
            season.append(season_i)
            season_label.append(season_i_label)
        
    #create xarray data array and save to netCDF format
    date_init = [nc_fc.time[0].values.astype(str)]
    out_arr = np.expand_dims(out_arr,axis=0) #add "rtime" dimension to add the date of the forecast init
    out_arr = xr.DataArray(out_arr, coords=[date_init,variable_fc,np.array((1,2,3)),season_label,nc_fc[lat_name[-1]],nc_fc[lon_name[-1]]], dims=['rtime','variable','tercile','season','y','x'], name='terciles')
    out_arr = out_arr.to_dataset()
    #set dimension attributes
    out_arr['rtime'].attrs['standard_name'] = 'forecast_reference_time'
    out_arr['rtime'].attrs['long_name'] = 'initialization date of the forecast'
    out_arr['tercile'].attrs['long_name'] = 'terciles in ascending order, 1 = lower, 2 = center, 3 = upper'
    out_arr['tercile'].attrs['tercile_period'] = nc_quantile.validation_period
    out_arr['tercile'].attrs['tercile_version'] = nc_quantile.version
    out_arr['variable'].attrs['long_name'] = 'name of the meteorological variable'
    out_arr['season'].attrs['long_name'] = 'season the forecast is valid for'
    #set variable attributes
    out_arr['terciles'].attrs['units'] = 'forecast probability ('+str(np.round(out_arr['terciles'].min().values,3))+' - '+str(np.round(out_arr['terciles'].max().values,3))+') for the most probable tercile, otherwise nan'
    out_arr['terciles'].attrs['detrended'] = detrended
    #set global attributes
    out_arr.attrs['model_system'] = model
    out_arr.attrs['model_version'] = version
    #out_arr.attrs['init'] = date_init
    out_arr.attrs['file_author'] = nc_quantile.author

    ##set chunking and save the file
    #out_arr = out_arr.chunk({"variable":1, "tercile":1, "season":1, "y":len(nc_fc[lat_name[-1]]), "x":len(nc_fc[lon_name[-1]])})
    encoding = dict(terciles=dict(chunksizes=(1, 1, 1, 1, len(nc_fc[lat_name[-1]]), len(nc_fc[lon_name[-1]])))) #https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
    savename = dir_forecast+'/terciles_'+model+version+'_init_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'_'+str(season_length)+'mon_dtr_'+detrended+'_refyears_'+str(years_quantile[0])+'_'+str(years_quantile[1])+'.nc'
    out_arr.to_netcdf(savename,encoding=encoding)

    #close all xarray objects
    lower_xr.close()
    upper_xr.close()
    seas_mean.close()
    nc_fc.close()
    out_arr.close()
    del(lower_xr,upper_xr,seas_mean,nc_fc,out_arr)
    
nc_quantile.close()
del(nc_quantile)

print('INFO: pred2tercile.py has been run successfully ! The netcdf output files have been stored at '+dir_forecast)
