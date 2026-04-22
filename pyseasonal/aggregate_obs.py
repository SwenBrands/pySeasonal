#!/usr/bin/env python

'''aggregates ERA5 data on ecmwf51 grid produced by Predictia or gridded PTI observations to monthly mean values compatible with the netCDF files produce by regrid_obs.py, both of which are then fed to get_skill_season.py'''

#load packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import pdb

from pyseasonal.utils.config import load_config

# INDICATE CONFIGURATION FILE ######################################

# configuration_file = 'config/config_for_aggregate_obs_Canarias.yaml'
# configuration_file = 'config/config_for_aggregate_obs_Canarias.yaml'
configuration_file = 'config/config_for_aggregate_obs_medcof.yaml'

####################################################################

# Load configuration
config = load_config(configuration_file)

#set input parameters for observational datasets to be regridded:
obs = config['obs'] #name of the observational / reanalysis dataset that will be regridded: 'era5', 'PTI-grid-v2'
agg_src = config['agg_src'] #'day' or 'mon': temporal aggregation of the observational input files, pertains to the <obs> loop indicated with <oo> below
variables = config['variables'] #variables to be regridded
variables_nc = config['variables_nc'] #variable names with the netCDF file
startyears_file = config['startyears_file'] # list of start years of the obs file as indicated in filename
endyears_file = config['endyears_file'] #list of corresponding end years
startyears_aggregation = config['startyears_aggregation'] #list of start years per variable to be regridded
endyears_aggregation = config['endyears_aggregation'] #list of start years per variable to be regridded
domain = config['domain'] #spatial domain the model data is available on. So far, this is just a label used in the output filename.
domain_label = config['domain_label'] #domain label used in file name
resolution = config['resolution'] #resolution shortcut used in the input variable names
grid_name = config['grid_name'] #here used as label to save the output netcdf file: ecmwf51 for 1 degree datasets, PTI-grid-v2 for downscaled datasets
int_method = config['int_method'] #here used as label to save the output netcdf file, ask Adri if this method was used, 'conservative_normed' or 'upscaled'

# Extract paths from configuration
paths = config['paths']
home = paths['home']
path_obs_base = paths['path_obs_base']
savepath_base = paths['savepath_base']

## EXECUTE #############################################################

#create output directory if it does not exist.
if os.path.isdir(savepath_base+'/'+obs) != True:
    os.makedirs(savepath_base+'/'+obs)

#Load observations, cut out years indicated in <years> and aggregate to monthly mean value. Then save to netcdf
for vv in np.arange(len(variables)):
    print('Processing '+variables[vv]+' from '+obs+' from '+str(startyears_aggregation[vv])+' to '+str(endyears_aggregation[vv]))
    if variables[vv] in ('t2m','tp','sst','z500','si10','ssrd','msl'):
        raise ValueError(variables[vv]+' are already available on monthly timescale in '+path_obs_base+' because they have been already donwloaded from CDS!')

    #define path to the file and loading function as a function of the variable (variables may come from different providers using distinct rules)
    if domain == 'medcof':
        if variables[vv] == 'fwi':
            path_obs_data = path_obs_base+'/'+obs.upper()+'_deprecated/data_derived/'+domain+'_'+resolution+'/day/'+variables[vv]+'12/'+variables[vv]+'_'+domain_label+'_'+resolution+'.nc'
            nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
        elif variables[vv] == 'pvpot':
            path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/day/'+variables[vv]+'/'+variables[vv]+'_'+domain_label+'_'+resolution+'_DM.nc'
            nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
        # elif variables[vv] in ('SPEI-3','GDD_S','GDD_W','CGDD_S','CGDD_W'):
        #     #path_obs_data = path_obs_base+'/'+obs+'/'+agg_src+'/'+domain+'_'+resolution+'/'+variables[vv]+'/'+variables[vv]+'_'+obs.upper()+'_'+agg_src+'_*.nc' #SPEI-3_ERA5_day_2021.nc
        #     path_obs_data = path_obs_base+'/'+obs.upper()+'_deprecated/data_derived/'+domain+'_'+resolution+'/'+agg_src+'/'+variables[vv] #SPEI-3_ERA5_day_2021.nc
        #     #get list of input files
        #     inputfiles_list = []
        #     for yy in np.arange(startyears_aggregation[vv],endyears_aggregation[vv]+1):
        #         dir_content = os.listdir(path_obs_data+'/'+str(yy))
        #         for fi in np.arange(len(dir_content)):
        #             inputfiles_list.append(path_obs_data+'/'+str(yy)+'/'+dir_content[fi])
        #     inputfiles_list = sorted(inputfiles_list)
        #     print('The following input files will be loaded and concatenated:')
        #     print(inputfiles_list)
        #     nc = xr.open_mfdataset(inputfiles_list, combine="by_coords")
        elif variables[vv] in ('SPEI-3','GDD_S','GDD_W','CGDD_S','CGDD_W','PVPOTm','FWIm','DD','SU','FD','ID','TR','pet-hargreaves','PRm','Rx1day','Rx5day','SSRDm','Tm','TNm','TXm','UAI','WSm'):
            path_obs_data = path_obs_base+'/'+obs.upper()+'/data_derived/'+domain+'_'+resolution+'/'+agg_src+'/'+variables[vv]+'/'+variables[vv]+'_'+obs.upper()+'_'+domain_label+'_'+resolution+'_'+agg_src+'_'+str(startyears_file[vv])+'-'+str(endyears_file[vv])+'.nc'
            nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
        else:
            raise ValueError('Variable '+variables[vv]+' is not yet supported by this script !')
    elif domain in ('Canarias','Iberia'):
        path_obs_data = path_obs_base+'/'+obs+'/data_derived_'+resolution+'/'+domain+'/'+agg_src+'/'+variables[vv]+'/'+variables[vv]+'_'+domain_label+'.nc'
        #remove resolution from the path in case the native PTI grid is to loaded
        if (domain == 'Canarias' and resolution == '2.5km') or (domain == 'Iberia' and resolution == '2.5km'):
            path_obs_data = path_obs_data.replace('_'+resolution,'')
            
        nc = xr.open_dataset(path_obs_data, decode_timedelta=False)
    else:
        raise ValueError('Unexpected value for <domain> !')

    # pdb.set_trace()
    nc = nc.rename({variables_nc[vv]:variables[vv]})

    #cut out target period
    dates = pd.DatetimeIndex(nc.time.values)
    years_ind = np.where((dates.year >= startyears_aggregation[vv]) * (dates.year <= endyears_aggregation[vv]))[0]
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
        var_units = nc[variables[vv]].units
    except:
        print('WARNING: Units for '+variables[vv]+' are missing and will be added now !')
        if variables[vv] in ('SU', 'FD', 'ID', 'TR', 'DD'):
            pdb.set_trace()
            var_units = 'day'
        elif variables[vv] in ('TXm', 'TNm', 'CGDD_W', 'CGDD_s'):
            pdb.set_trace()
            var_units = 'degC'
        elif variables[vv] in ('PRm','Rx1day','Rx5day'):
            pdb.set_trace()
            var_units = 'kg m-2'
        elif variables[vv] in ('SSRDm'):
            pdb.set_trace()
            var_units = 'W m-2'
        elif variables[vv] in ('pet_hargreaves'):
            pdb.set_trace()
            var_units = 'kg m-2 s-1'
        elif variables[vv] in ('UAI','FWIm','fwi'):
            var_units = 1
        elif variables[vv] in ('WSm'):
            pdb.set_trace()
            var_units = 'm s-1'
        elif variables[vv] in ('SPEI-3'):
            pdb.set_trace()
            var_units = 1
        elif variables[vv] in ('PVPOTm','pvpot'):
            # pdb.set_trace()
            var_units = 1
        else:
            pdb.set_trace()

        print('WARNING: Setting unit for '+variables[vv]+' to '+str(var_units))

    nc[variables[vv]].attrs['units'] = var_units

    # nc.x.attrs('standard_name') = 'longitude'
    # nc.x.attrs('long_name') = 'longitude'
    # nc.x.attrs('units') = 'degrees_east'
    # nc.y.attrs('standard_name') = 'latitude'
    # nc.y.attrs('long_name') = 'latitude'
    # nc.y.attrs('units') = 'degrees_north'
    savepath = savepath_base+'/'+obs+'/'+variables[vv]+'_mon_'+obs+'_on_'+grid_name+'_grid_'+int_method+'_'+domain+'_'+str(startyears_aggregation[vv])+'_'+str(endyears_aggregation[vv])+'.nc'
    nc.astype('float32').to_netcdf(savepath)
    nc.close()
    del(nc)

print('INFO: aggregate_obs.py has been run successfully! The output nc files containing the re-organized data is '+savepath)

