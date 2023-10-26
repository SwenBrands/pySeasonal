#!/usr/bin/env python

'''regrids observational data from the datasest defined in <obs> to grid defined by <model> and <version>, optionally taking into account the land sea-masks of these datasets (this option is not working yet).
The GCM data was obtained from https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview'''

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
obs = ['era5'] #name of the observational / reanalysis dataset that will be regridded
agg_src = ['mon'] #temporal aggregation of the observational input files, pertains to the <obs> loop indicated with <oo> below
startyear_file = [1940] #start year of the obs file as indicated in filename
endyear_file = [2022] #corresponding end year
variables = ['tp','t2m'] #variables to be regridded
years = [1981,2022] #years to be regridded
int_method = 'conservative_normed' #'conservative_normed', interpolation method used by xesmf

#set input parameters for model datasets, only used to get the land-sea mask of the models listed in <model>
model = ['ecmwf'] #seasonal forecast model
version = ['51'] #and version thereof; pertains to <model> loop indicated with <mm> below
lsm_thresh = [0.5] #fraction above which the grid box is considered land in the gcm (the value is continuous between 0 and 1 in ecmwf 51)
product = 'hindcast' #considered product
home = os.getenv('HOME')

#set MEDCOF domain
domain = 'medcof' #spatial domain the model data is available on. So far, this is just a label used in the output filename.
latlim_o = [13.5,53.5] #this is the domain used for observations. It is slightly larger than the model grid in order to have sufficient observational gridpoints around a given model grid-box for interpolation
lonlim_o = [-20.5,51.5] #the datasets come on 0 to 360 degrees longitude and are rolled to 0 <= longitude < 360 degrees
latlim_m = [14,53]
lonlim_m = [-20,51]

#set basic path structure for observations and land-sea masks from the models
path_obs_base = home+'/datos/OBSData' #base path upon which the final path to the obs data is constructed
path_gcm_base = home+'/datos/GCMData/seasonal-original-single-levels' #here, the land sea masks of all models and versions thereof are located
savepath_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/obs/regridded' #The nc files generated here, containing observations regridded to the model grid, are stored in this directory

## EXECUTE #############################################################
for oo in np.arange(len(obs)):
    #get land-sea mask from observational dataset
    path_obs_lsm = path_obs_base+'/'+obs[oo]+'/lsm/lsm_'+obs[oo]+'.nc'
    nc_obs_lsm = roll_and_cut(xr.open_dataset(path_obs_lsm),lonlim_o,latlim_o)
    land_obs = nc_obs_lsm.lsm.values == 1 #value is binary in era5
    sea_obs = nc_obs_lsm.lsm.values == 0
    nc_obs_lsm.close()    
   
    for mm in np.arange(len(model)):
        
        #get land-sea mask from GCM dataset        
        # path_gcm_lsm = path_gcm_base+'/lsm/lsm_'+model[mm]+version[mm]+'.nc' #comes with .5 instead of .0 resolution bins in lat an lon which does not agree with the ecmwf51 downloaded to lustre
        # nc_gcm_lsm = roll_and_cut(xr.open_dataset(path_gcm_lsm),lonlim_m,latlim_m)
        # land_gcm = nc_gcm_lsm.lsm.values > lsm_thresh[mm]
        # sea_gcm = nc_gcm_lsm.lsm.values <= lsm_thresh[mm]
        # nc_gcm_lsm.close()
        
        #alternatively load GCM grid from any of the netCDF files containing meteorological varialbes stored on lustre, the land-sea mask is not loaded if this option is followed
        path_gcm_lsm = path_gcm_base+'/'+domain+'/hindcast/tas/'+model[mm]+'/'+version[mm]+'/198101/seasonal-original-single-levels_'+domain+'_hindcast_tas_'+model[mm]+'_'+str(version[mm])+'_198101.nc'
        nc_gcm_lsm = xr.open_dataset(path_gcm_lsm)
        
        #Then load observations, cut out years indicated in <years>, interpolate obs variable to the gcm grid and save as netCDF file
        for vv in np.arange(len(variables)):
            path_obs_data = path_obs_base+'/'+obs[oo]+'/'+agg_src[oo]+'/'+obs[oo]+'_mon_'+variables[vv]+'_'+str(startyear_file[oo])+'_'+str(endyear_file[oo])+'.nc'
            nc_obs_data = roll_and_cut(xr.open_dataset(path_obs_data),lonlim_o,latlim_o)
            obs_dates = pd.DatetimeIndex(nc_obs_data.time.values)
            yearbool = (obs_dates.year >= years[0]) & (obs_dates.year <= years[1])
            nc_obs_data = nc_obs_data.isel(time=yearbool)
            
            #regrid without mask
            print('INFO: Regridding '+variables[vv]+' from '+obs[oo]+' on '+model[mm]+' '+version[mm]+' grid using '+int_method+' method from xesmf...')
            #regridder = xesmf.Regridder(nc_obs_data,nc_gcm_lsm.drop_dims('number'), method=int_method)
            regridder = xesmf.Regridder(nc_obs_data,nc_gcm_lsm.drop_dims('member'), method=int_method)
            nc_obs_int = regridder(nc_obs_data, keep_attrs=True)
            savepath = savepath_base+'/'+obs[oo]+'/'+variables[vv]+'_'+agg_src[oo]+'_'+obs[oo]+'_on_'+model[mm]+version[mm]+'_grid_'+int_method+'_'+domain+'_'+str(years[0])+'_'+str(years[1])+'.nc'
            print('INFO: Writing '+obs[oo]+' '+variables[vv]+' data regridded to '+model[mm]+version[mm]+' grid on '+domain+' domain with '+int_method+' method to: '+savepath)
            nc_obs_int.to_netcdf(savepath)
            nc_obs_int.close()
            nc_obs_data.close()
print('INFO: regrid_obs.py has been run successfully! The output nc files containing the observational data interpolated to a model grid can be found in '+savepath_base)

