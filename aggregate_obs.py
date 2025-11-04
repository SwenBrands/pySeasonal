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
import pdb
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters for observational datasets to be regridded:
obs = 'PTI-grid-v2' #name of the observational / reanalysis dataset that will be regridded: 'era5', 'PTI-grid-v2'
agg_src = 'mon' #'day' or 'mon': temporal aggregation of the observational input files, pertains to the <obs> loop indicated with <oo> below
startyear_file = 1981 #start year of the obs file as indicated in filename
endyear_file = 2022 #corresponding end year
#variables = ['pvpot','SPEI-3','fwi'] #variables to be regridded
#variables_nc = ['pvpot','SPEI-3','FWI'] #variable names with the netCDF file
variables = ['FD','SU','TR'] #variables to be regridded
variables_nc = ['FD','SU','TR'] #variable names with the netCDF file
years = [1981,2022] #years to be regridded
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files

# #set MEDCOF domain
# domain = 'medcof' #spatial domain the model data is available on. So far, this is just a label used in the output filename.
# resolution = '1degree' #resolution shortcut used in the input variable names

#set Iberia or Canarias domains
domain = 'Canarias' #spatial domain the model data is available on. So far, this is just a label used in the output filename.
resolution = '5km' #resolution shortcut used in the input variable names

grid_name = 'PTI-grid-v2' #here used as label to save the output netcdf file: ecmwf51 for 1 degree datasets, PTI-grid-v2 for downscaled datasets
int_method = 'upscaled' #here used as label to save the output netcdf file, ask Adri if this method was used, 'conservative_normed' or 'upscaled'

## EXECUTE #############################################################
#set basic path structure for observations and land-sea masks from the models
if file_system == 'lustre' and domain == 'medcof':
    home = '/lustre/gmeteo/PTICLIMA'
    path_obs_base = home+'/DATA/REANALYSIS' #base path upon which the final path to the obs data is constructed
    savepath_base = home+'/Results/seasonal/obs/regridded' #This is the common directory for regridded monthly observations; what is placed here will be processed by get_skill_season.py
elif file_system == 'lustre' and domain == 'Canarias':  # /lustre/gmeteo/PTICLIMA/DATA/OBSERVATIONS/PTI-grid-v2/data_derived_5km/Canarias/mon
    home = '/lustre/gmeteo/PTICLIMA'
    path_obs_base = home+'/DATA/OBSERVATIONS' #base path upon which the final path to the obs data is constructed
    savepath_base = home+'/Results/seasonal/obs/regridded' #This is the common directory for regridded monthly observations; what is placed here will be processed by get_skill_season.py
else:
    raise ValueError('Unknown entry for <file_system> input parameter !')
print('The file system is '+file_system+'...')

#create output directory if it does not exist.
if os.path.isdir(savepath_base+'/'+obs) != True:
    os.makedirs(savepath_base+'/'+obs)

#Load observations, cut out years indicated in <years> and aggregate to monthly mean value. Then save to netcdf
for vv in np.arange(len(variables)):
    print('Processing '+variables[vv]+' from '+obs+' for the years '+str(years)+' on file system '+file_system+'...')
    if variables[vv] in ('t2m','tp','sst','z500','si10','ssrd','msl'):
        raise ValueError(variables[vv]+' are already available on monthly timescale in '+path_obs_base+' because they have been already donwloaded from CDS!')
    
    #define path to the file and loading function as a function of the variable (variables may come from different providers using distinct rules)
    if domain == 'medcof':
        if variables[vv] == 'fwi':
            path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/'+variables[vv]+'12/'+variables[vv]+'_'+domain+'_'+resolution+'.nc'
            nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
        elif variables[vv] == 'pvpot':
            path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/'+variables[vv]+'/'+variables[vv]+'_'+domain+'_'+resolution+'_DM.nc'
            nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
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
            raise ValueError('Variable '+variables[vv]+' is not yet supported by this script !')
    elif domain in ('Canarias','Iberia'):
        # path_obs_data = path_obs_base+'/'+obs[0:3].upper()+obs[3:]+'/data_derived_'+resolution+'/'+domain[0].upper()+domain[1:]+'/'+agg_src+'/'+variables[vv]+'/'+variables[vv]+'_'+domain[0:3]+'.nc'
        path_obs_data = path_obs_base+'/'+obs+'/data_derived_'+resolution+'/'+domain+'/'+agg_src+'/'+variables[vv]+'/'+variables[vv]+'_'+domain.lower()[0:3]+'.nc'
        nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
    else:
        valueError('Unexpected value for <domain> !')
    
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

    # add units
    # add exception for pvpot, which currently comes without units
    try:
        print(nc[variables[vv]].units)
    except:
        print('WARNING: Units for '+variables[vv]+' are missing and will be added now !')
        if variables[vv] == 'pvpot':
            units_content = 'index value ranging from 0 to 1'
            print('WARNING: Setting unit for '+variables[vv]+' to '+units_content)
            nc[variables[vv]].attrs['unit'] = units_content
        # else:
        #     # pdb.set_trace() #for some reason, xarray does not read the <units> attribute from netcdf files created by climate4R neither does it permit to store a new <units> attribute. As a work-around, <longname> is read instead and saved as <unit>; correct this in future versions !
        #     units_content = nc[variables[vv]].longname
 
    # nc.x.attrs('standard_name') = 'longitude'
    # nc.x.attrs('long_name') = 'longitude'
    # nc.x.attrs('units') = 'degrees_east'
    # nc.y.attrs('standard_name') = 'latitude'
    # nc.y.attrs('long_name') = 'latitude'
    # nc.y.attrs('units') = 'degrees_north'
    savepath = savepath_base+'/'+obs+'/'+variables[vv]+'_mon_'+obs+'_on_'+grid_name+'_grid_'+int_method+'_'+domain+'_'+str(years[0])+'_'+str(years[1])+'.nc'
    nc.to_netcdf(savepath)
    nc.close()
    del(nc)
    
print('INFO: aggregate_obs.py has been run successfully! The output nc files containing the re-organized data is '+savepath)

