#!/usr/bin/env python

'''transforms NOAA's Oceanic Niño index downloaded in netCDF format from 'https://psl.noaa.gov/data/correlation/oni.nc' to categorical data indicating
Niño (1), Niña (2) and neutral (0) conditions indices and stores the results in netcdf format. Niño or Niño conditions are declared if the ONI index (which itself
is a 3-months running-mean values) is above or below +0.5 or -0.5 during 5 consecutive months, following https://www.ncei.noaa.gov/access/monitoring/enso/sst/'''

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

#set input parameters
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files
url_index = 'https://psl.noaa.gov/data/correlation/oni.nc' #can be used optionally to read the online file directly; currently does not work !
enso_threshold = 0.5 #magnitude of the ONI index above which Niño or Niña conditions are declared. Is used symmetrically around 0
input_filename = 'oni.nc'

#set basic path structure for observations and gcms
if file_system == 'myLaptop':
    home = os.getenv('HOME')
    rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal'
    #dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation/'+vers
elif file_system == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    rundir = home+'/Scripts/SBrands/pyPTIclima/pySeasonal'
    obsdir = home+'/DATA/OBSERVATIONS/ONI/data/'
    dir_netcdf = home+'/Results/seasonal/indices'
else:
    raise Exception('ERROR: unknown entry for <file_system> input parameter!')

## EXECUTE #####################################################################
#create output file directory if necessary
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

nc = xr.open_dataset(obsdir+'/'+input_filename)

#get booleans for warm, cold and neutral index values
warm = (nc.value > enso_threshold).astype(int)
cold = (nc.value < enso_threshold*-1).astype(int)
neutral = (np.abs(nc.value) <= enso_threshold).astype(int) #currently not used

#get rolling 5 months sum
warm = warm.rolling(time=5,min_periods=5,center=False).sum()
cold = cold.rolling(time=5,min_periods=5,center=False).sum()
nanind = np.isnan(warm) #index for nan time instances

#find El Niño and La Niña events
nino = warm == 5
nina = cold == 5

#get an enso indes time series with 0 = neutral, 1 = El Niño and 2 = La Niña
index = warm.copy().rename('oni2enso')
index[:] = 0
index[nino] = 1
index[nina] = 2
index[nanind] = np.nan

#set index attributes
index.attrs['name'] = 'oni2enso'
index.attrs['standard_name'] = 'oni2enso'
index.attrs['long_name'] = 'ENSO index based on Oceanic Niño index'
index.attrs['source'] = 'https://psl.noaa.gov/data/correlation/oni.nc'
index.attrs['Description'] = 'ONI based ENSO index as used by NOAA; Niño and Niña conditions are issued if the ONI index is above or below +-0.5 during 5 consecutive months.'
index.attrs['units'] = '0 = neutral, 1 = Niño, 2 = Niña conditions'
index.attrs['creator'] = 'Swen Brands, brandssf@ifca.unican.es'

savename = dir_netcdf+'/oni2enso_'+nc.time.values[0].astype(str)[0:7].replace('-','')+'_'+nc.time.values[-1].astype(str)[0:7].replace('-','')+'.nc'
index.to_netcdf(savename)
index.close()
nc.close()
warm.close()
cold.close()
neutral.close()




