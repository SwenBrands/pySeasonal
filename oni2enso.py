#!/usr/bin/env python

'''transforms NOAA's Oceanic Niño index downloaded in txt format from <url_index> (is set below) to categorical data indicating
Niño (1), Niña (2) and neutral (0) conditions indices and stores the results in netcdf format. Niño or Niño conditions are declared if the ONI index (which itself
is a 3-months running-mean values) is above or below +0.5 or -0.5 during <window> consecutive months, where <window> = 5 following NOAA's definition at https://www.ncei.noaa.gov/access/monitoring/enso/sst/'''

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
import pdb

#set input parameters
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files
url_index = 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt' #URL of the online txt file
enso_threshold = 0.5 #magnitude of the ONI index above which Niño or Niña conditions are declared. Is used symmetrically around 0
window = 1 #number of consecutive months during which the 3-month mean SST anomaly values provided by NOAA must surpass the magnitude of the <enso_threshold> in order to issue a Niño or Niña event 

#set basic path structure for observations and gcms
if file_system == 'myLaptop':
    home = os.getenv('HOME')
    rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal'
    dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation/'
elif file_system == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    rundir = home+'/Scripts/SBrands/pyPTIclima/pySeasonal' #the running directory; the script will be run in this directory
    dir_netcdf = home+'/Results/seasonal/indices' #the output file directory
else:
    raise Exception('ERROR: unknown entry for <file_system> input parameter!')

## EXECUTE #####################################################################
#create output file directory if necessary
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

df = pd.read_csv(url_index,delim_whitespace=True) #read the online index from CPC
time = [df['YR'].values[ii].astype('str').zfill(2)+'-'+df['MON'].values[ii].astype('str').zfill(2)+'-01' for ii in np.arange(df.shape[0])]
time =  pd.DatetimeIndex(time)
nc = xr.DataArray(df['ANOM'].values, coords=[time], dims='time', name = 'oni')
nc = nc.rolling(time=3,min_periods=3,center=False).mean() #calculate 3-months running mean values

warm = (nc > enso_threshold).astype(int)
cold = (nc < enso_threshold*-1).astype(int)
neutral = (np.abs(nc) <= enso_threshold).astype(int) #currently not used

#get rolling <window> months sum
warm = warm.rolling(time=window,min_periods=window,center=False).sum()
cold = cold.rolling(time=window,min_periods=window,center=False).sum()
nanind = np.isnan(warm) #index for nan time instances

#find El Niño and La Niña events
nino = warm == window
nina = cold == window

#get an enso indes time series with 0 = neutral, 1 = El Niño and 2 = La Niña
index = warm.copy().rename('oni2enso')
index[:] = 0
index[nino] = 1
index[nina] = 2
index[nanind] = np.nan

#set index attributes
index.attrs['name'] = 'oni2enso'
index.attrs['standard_name'] = 'oni2enso'
index.attrs['long_name'] = 'ENSO index based on Oceanic Niño Index, defined as 3-month mean value of the detrended monthly SST anomalies in the Niño 3.4 region.'
index.attrs['source'] = url_index
index.attrs['Description'] = 'ONI based ENSO index as used by NOAA; Niño and Niña conditions are issued if the ONI index is above or below +-0.5 during '+str(window)+' consecutive months.'
index.attrs['units'] = 'categorical, 0 = neutral, 1 = Niño, 2 = Niña conditions'
index.attrs["cell_methods"] = 'time: sum ('+str(window)+' months)'
index.attrs['creator'] = 'Swen Brands, brandssf@ifca.unican.es'

savename = dir_netcdf+'/oni2enso_'+nc.time.values[0].astype(str)[0:7].replace('-','')+'_'+nc.time.values[-1].astype(str)[0:7].replace('-','')+'.nc'
index.to_netcdf(savename)
index.close()
nc.close()
warm.close()
cold.close()
neutral.close()




