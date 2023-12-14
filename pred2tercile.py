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
import pdb as pdb #then type <set_trace()> at a given line in the code below
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters
quantile_version = '1g' #version number of the quantiles file used here
model = 'ecmwf' #interval between meridians and parallels
version = '51'

year_init = [2023,2023] #a list containing the years the forecast are initialized on, will be looped through with yy
month_init = [11,12] #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

years_quantile = [1981,2022] #years used to calculate the quantiles with get_skill_season.py
season_length = 3 #length of the season, e.g. 3 for DJF, JFM, etc.
detrended = 'no'

variable_qn = ['msl','t2m','tp','si10','ssrd'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>
variable_fc = ['psl','tas','pr','sfcWind','rsds'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>

datatype = 'float32' #data type of the variables in the output netcdf files
domain = 'medcof' #spatial domain
detrended = 'no' #yes or no, linear detrending of the gcm and obs time series prior to validation
nr_mem = [25] #considered ensemble members, not yet in use !

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

#set basic path structure for observations and gcms
home = os.getenv('HOME')
rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal'
dir_quantile = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation'
dir_forecast = home+'/datos/tareas/proyectos/pticlima/seasonal/results/forecast'
gcm_store = 'F' #laptop, F or extdisk2
product = 'forecast'

## EXECUTE #############################################################
quantile_threshold = [0.33,0.67]

#set path to input gcm files
if gcm_store == 'laptop':
    path_gcm_base = home+'/datos/GCMData/seasonal-original-single-levels/'+domain # head directory of the source files
elif gcm_store == 'F':
    path_gcm_base = '/media/swen/F/datos/GCMData/seasonal-original-single-levels/'+domain # head directory of the source files
elif gcm_store == 'extdisk2':
    path_gcm_base = '/media/swen/ext_disk2/datos/GCMData/seasonal-original-single-levels/'+domain # head directory of the source files
else:
    raise Exception('ERROR: unknown entry for <path_gcm_base> !')
print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

#load the quantiles file
filename_quantiles = dir_quantile+'/quantiles_pticlima_'+domain+'_'+str(years_quantile[0])+'_'+str(years_quantile[1])+'_v'+quantile_version+'.nc'
nc_quantile = xr.open_dataset(filename_quantiles)

for yy in np.arange(len(year_init)):
    for vv in np.arange(len(variable_fc)):
        #load forecast file 
        filename_forecast = path_gcm_base+'/'+product+'/'+variable_fc[vv]+'/'+model+'/'+version+'/'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'/seasonal-original-single-levels_'+domain+'_'+product+'_'+variable_fc[vv]+'_'+model+'_'+version+'_'+str(year_init[yy])+str(month_init[yy]).zfill(2)+'.nc'
        nc_fc = xr.open_dataset(filename_forecast)

        #transform GCM variables and units, if necessary
        nc_fc, file_valid = transform_gcm_variable(nc_fc,variable_fc[vv],variable_qn[vv],model,version)
        #check whether there is a problem with the variable units as revealed in transform_gcm_variable()
        if file_valid == 0:
            raise Exception('ERROR: There is a problem with the expected variable units in '+filename_forecast+' !')

        # if (variable_fc[vv] == 'tas') & (model == 'ecmwf') & (version == '51'):
            # print('Info: Adding 2x273.15 to '+variable_fc[vv]+' data from '+model+version+' to correct Predictia workflow error and then transform into Kelvin.')
            # #nc_fc[variable_fc[vv]].values = nc_fc[variable_fc[vv]].values+273.15
            # nc_fc[variable_fc[vv]][:] = nc_fc[variable_fc[vv]]+273.15+273.15
            # nc_fc[variable_fc[vv]].attrs['units'] = 'daily mean '+variable_qn[vv]+' in Kelvin'
        # elif (variable_fc[vv] in ('pr','rsds')) & (model == 'ecmwf') & (version == '51'):
            # print('Info: Disaggregate '+variable_fc[vv]+' accumulated over the '+str(len(nc_fc.time))+' days forecast period from '+model+version+' to daily sums.')
            # vals_disagg = np.diff(nc_fc[variable_fc[vv]].values,n=1,axis=0)
            # shape_disagg = vals_disagg.shape
            # add_day = np.expand_dims(vals_disagg[-1,:,:,:],axis=0) #get last available difference
            # #add_day = np.zeros((1,shape_disagg[1],shape_disagg[2],shape_disagg[3]))
            # #add_day[:] = np.nan
            # vals_disagg = np.concatenate((vals_disagg,add_day),axis=0)
            # nc_fc[variable_fc[vv]][:] = vals_disagg
            # nc_fc[variable_fc[vv]].attrs['units'] = 'daily accumulated '+variable_qn[vv]+' in '+nc_fc[variable_fc[vv]].attrs['units']
        # else:
            # print('Info: No data transformation is applied for '+variable_fc[vv]+' data from '+model+version+'.')
        
        #get forecast seasons from file
        dates_fc = pd.DatetimeIndex(nc_fc.time.values)
        months_fc_uni = dates_fc[15::30].month #get unique forecast month in the right order, i.e. as appears in the file
        season = []
        season_label = []
        season_start_month = np.arange(len(months_fc_uni)-season_length+1) #index of the first month of each 3-month period
        
        #init final output array
        if vv == 0:
            out_arr = np.zeros((len(variable_fc),len(quantile_threshold)+1,len(season_start_month),len(nc_fc.y),len(nc_fc.x)),dtype=datatype)
            out_arr[:] = np.nan
        for mo in season_start_month:
            season_i = months_fc_uni[mo:mo+season_length].to_list()
            season_i_label = assign_season_label(season_i)
            season_ind = np.where(np.isin(dates_fc.month,season_i))[0]
            seas_mean = nc_fc[variable_fc[vv]].isel(forecast_time=season_ind).mean(dim='forecast_time')
            #get the ensemble quantiles for this season and leadtime (note that the season and leadtime have the same index !)
            lower_xr = nc_quantile.sel(detrended=detrended,model=model+version,quantile_threshold=quantile_threshold[0],variable=variable_qn[vv],season=season_i_label).quantile_ensemble.isel(lead=mo) #is a 2D xarray data array
            upper_xr = nc_quantile.sel(detrended=detrended,model=model+version,quantile_threshold=quantile_threshold[1],variable=variable_qn[vv],season=season_i_label).quantile_ensemble.isel(lead=mo)
            lower_np = np.tile(lower_xr.values,(seas_mean.shape[0],1,1))
            upper_np = np.tile(upper_xr.values,(seas_mean.shape[0],1,1))
            
            #True of False for either of the 3 categories
            upper_ind = seas_mean > upper_np
            center_ind = (seas_mean >= lower_np) & (seas_mean <= upper_np)
            lower_ind = seas_mean < lower_np
            #sum members in each category
            nr_mem = upper_ind.shape[0]
            upper_nr = upper_ind.sum(dim='member')/nr_mem
            center_nr = center_ind.sum(dim='member')/nr_mem
            lower_nr = lower_ind.sum(dim='member')/nr_mem
            #stack and turn to numpy format
            terciles = np.stack((lower_nr.values,center_nr.values,upper_nr.values),axis=0)
            terciles_nan = np.zeros((terciles.shape))
            terciles_nan[:] = np.nan
            #get index of the tercile having the maximum forecast probability at each grid box; save this probability at the given tercile; at the other terciles the nan value will be kept.
            for ii in np.arange(terciles.shape[1]):
                for jj in np.arange(terciles.shape[2]):
                    maxind = np.argmax(terciles[:,ii,jj])
                    terciles_nan[maxind,ii,jj] = terciles[maxind,ii,jj]
            #terciles_nan[maxprob_ind] = terciles[maxprob_ind,:,:]
            out_arr[vv,:,mo,:,:] = terciles_nan
            season.append(season_i)
            season_label.append(season_i_label)
        
    #create xarray data array and save to netCDF format
    date_init = [nc_fc.time[0].values.astype(str)]
    out_arr = np.expand_dims(out_arr,axis=0) #add "rtime" dimension to add the date of the forecast init
    out_arr = xr.DataArray(out_arr, coords=[date_init,variable_fc,np.array((1,2,3)),season_label,nc_fc.y,nc_fc.x], dims=['rtime','variable','tercile','season','y','x'], name='terciles')
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
    #out_arr = out_arr.chunk({"variable":1, "tercile":1, "season":1, "y":len(nc_fc.y), "x":len(nc_fc.x)})
    encoding = dict(terciles=dict(chunksizes=(1, 1, 1, 1, len(nc_fc.y), len(nc_fc.x)))) #https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
    savename = dir_forecast+'/terciles_'+model+version+'_init_'+str(year_init[yy])+str(month_init[yy])+'_'+str(season_length)+'mon_dtr_'+detrended+'_refyears_'+str(years_quantile[0])+'_'+str(years_quantile[1])+'.nc'
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

print('INFO: pred2tercile.py has been run successfully !')
