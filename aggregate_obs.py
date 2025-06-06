#!/usr/bin/env python

'''aggregates ERA5 data on ecmwf51 grid produced by Predictia to monthly mean values compatible with the netCDF files produce by regrid_obs.py, both of which are then fed to get_skill_season.py'''

#load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import os
import xesmf
import pandas as pd
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters for observational datasets to be regridded
obs = 'era5' #name of the observational / reanalysis dataset that will be regridded
agg_src = 'day' #temporal aggregation of the observational input files, pertains to the <obs> loop indicated with <oo> below
startyear_file = 1981 #start year of the obs file as indicated in filename
endyear_file = 2022 #corresponding end year
#variables = ['pvpot','SPEI-3','fwi'] #variables to be regridded
#variables_nc = ['pvpot','SPEI-3','FWI'] #variable names with the netCDF file
variables = ['pvpot'] #variables to be regridded
variables_nc = ['pvpot'] #variable names with the netCDF file
years = [1981,2022] #years to be regridded
int_method = 'conservative_normed' #'conservative_normed', interpolation method used to regrid the observations which already are provided on the model grid
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files

#set MEDCOF domain
domain = 'medcof' #spatial domain the model data is available on. So far, this is just a label used in the output filename.
resolution = '1degree' #resolution shortcut used in the input variable names

model = 'ecmwf' #here used as label to save the output netcdf file
version = '51' #here used as label to save the output netcdf file
int_method = 'conservative_normed' #here used as label to save the output netcdf file, ask Adri if this method was used

## EXECUTE #############################################################
#set basic path structure for observations and land-sea masks from the models
if file_system == 'myLaptop':
    home = os.getenv('HOME')
    path_obs_base = home+'/datos/OBSData' #base path upon which the final path to the obs data is constructed
    savepath_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/obs/regridded' #This is the common directory for regridded monthly observations; what is placed here will be processed by get_skill_season.py
elif file_system == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    path_obs_base = home+'/DATA/REANALYSIS' #base path upon which the final path to the obs data is constructed
    savepath_base = home+'/Results/seasonal/obs/regridded' #This is the common directory for regridded monthly observations; what is placed here will be processed by get_skill_season.py
else:
    raise Exception('ERROR: unknown entry for <file_system> input parameter !')
print('The file system is '+file_system+'...')

#create output directory if it does not exist.
if os.path.isdir(savepath_base+'/'+obs) != True:
    os.makedirs(savepath_base+'/'+obs)

#Load observations, cut out years indicated in <years> and aggregate to monthly mean value. Then save to netcdf
for vv in np.arange(len(variables)):
    print('Processing '+variables[vv]+' from '+obs+' for the years '+str(years)+' on file system '+file_system+'...')
    if variables[vv] in ('t2m','tp','sst','z500','si10','ssrd','msl'):
        raise Expcetion('ERROR: '+variables[vv]+' are already available on monthly timescale in '+path_obs_base+' because they have been already donwloaded from CDS!')
    
    #define path to the file and loading function as a function of the variable (variables may come from different providers using distinct rules)
    if variables[vv] == 'fwi':
        path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/'+variables[vv]+'12/'+variables[vv]+'_'+domain+'_'+resolution+'.nc'
        nc = xr.open_dataset(path_obs_data)
    elif variables[vv] == 'pvpot':
        path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/'+variables[vv]+'/'+variables[vv]+'_'+domain+'_'+resolution+'_DM.nc'
        nc = xr.open_dataset(path_obs_data)
    elif variables[vv] == 'SPEI-3':
        #path_obs_data = path_obs_base+'/'+obs+'/'+agg_src+'/'+domain+'_'+resolution+'/'+variables[vv]+'/'+variables[vv]+'_'+obs.upper()+'_'+agg_src+'_*.nc' #SPEI-3_ERA5_day_2021.nc
        path_obs_data = path_obs_base+'/'+obs.upper()+'/data_masked/'+domain+'_'+resolution+'/'+variables[vv] #SPEI-3_ERA5_day_2021.nc
        #get list of input files
        inputfiles_list = []
        for yy in np.arange(years[0],years[1]+1):
            dir_content = os.listdir(path_obs_data+'/'+str(yy))
            for fi in np.arange(len(dir_content)):
                inputfiles_list.append(path_obs_data+'/'+str(yy)+'/'+dir_content[fi])
        print('The following input files will be loaded and concatenated:')
        print(inputfiles_list)
        nc = xr.open_mfdataset(inputfiles_list)
    else:
        raise Excpetion('ERROR: variable '+variables[vv]+' is not yet supported by this script !')
    
    nc = nc.rename({variables_nc[vv]:variables[vv]})
        
    #cut out target period
    dates = pd.DatetimeIndex(nc.time.values)
    years_ind = np.where((dates.year >= years[0]) & (dates.year <= years[-1]))[0]
    nc = nc.isel(time=years_ind)
    dates = pd.DatetimeIndex(nc.time.values) #retrieve dates form the time-reduced xr dataset
    #calculate monthly mean values
    print('INFO: calculating monthly mean values for the period '+str(dates[0])+' to '+str(dates[-1])+'. This may take a while...')
    #nc[variables[vv]] = nc[variables[vv]].resample(time="1MS").mean(dim='time') #retains an xr data array
    nc = nc.resample(time="1MS").mean(dim='time') #retains an xr dataset
    
    ##bring format to CDS standard for monthly mean values and save to netcdf format
    nc = nc.reindex(lat=list(reversed(nc.lat))) #brings latitudes to descending order
    nc = nc.rename_dims(lon='x',lat='y') #note that this does not rename the indices !
    nc = nc.rename_vars(lon='x',lat='y')
    # add exception for pvpot, which currently comes without units
    if variables[vv] == 'pvpot':
        try:
            print(nc[variables[vv]].units)
        except:
            print('Warning: Units for '+variables[vv]+' are missing and will be added now !')
            nc[variables[vv]].attrs['units'] = 'index value ranging from 0 to 1'
    # nc.x.attrs('standard_name') = 'longitude'
    # nc.x.attrs('long_name') = 'longitude'
    # nc.x.attrs('units') = 'degrees_east'
    # nc.y.attrs('standard_name') = 'latitude'
    # nc.y.attrs('long_name') = 'latitude'
    # nc.y.attrs('units') = 'degrees_north'
    savepath = savepath_base+'/'+obs+'/'+variables[vv]+'_mon_'+obs+'_on_'+model+version+'_grid_'+int_method+'_'+domain+'_'+str(years[0])+'_'+str(years[1])+'.nc'
    nc.to_netcdf(savepath)
    nc.close()
    del(nc)
    
print('INFO: aggregate_obs.py has been run successfully! The output nc files containing the daily observational data aggregated to monthly mean values can be found in '+savepath_base)

