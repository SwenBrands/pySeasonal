#!/usr/bin/env python

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
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters
#season = ['DJF','MAM','JJA','SON'] #months of the season to be evaluated
season_label = ['NDF','DJF']
season = [[11,12,1],[12,1,2]] #[[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
lead = [[0,1,2],[1,2,3]] #[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.
model = ['ecmwf'] #interval between meridians and parallels
version = ['51']
obs = ['era5']
years_model = [[1981,2022]] #years used in label of model netCDF file, refers to the first and the last year of the monthly model inits
years_obs = [[1981,2022]] #years used in label of obs netCDF file
variables = ['tp'] #['tp','t2m'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>
corr_outlier = 'no'
domain = 'medcof' #spatial domain
use_effn = 'yes' #use effective sample size to calculate the p-value of the Pearson and Spearman correlation coefficient.
detrending = 'no' #yes or no, linear detrending of the gcm and obs time series prior to validation

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

#set basic path structure for observations and gcms
home = os.getenv('HOME')
rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal'
path_obs_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/obs/regridded'
path_gcm_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/gcm/aggregated'
dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation'
path_figures = home+'/datos/tareas/proyectos/pticlima/seasonal/figures'

## EXECUTE #############################################################
#map results
lons = pearson_r.x.values
lats = pearson_r.y.values
#This is the map projection we want to plot *onto*, see https://docs.xarray.dev/en/stable/examples/visualization_gallery.html
map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
titlelabel = 'Temporal correlation for '+model[mm]+version[mm]+' ensmean vs. '+obs[oo]+' '+variables[vv]+' '+str(years_model[mm][0])+'-'+str(years_model[mm][-1])+' '+str(season[sea])
savelabel = path_figures+'/'+variables[vv]+'/pearson_effn'+'_'+use_effn+'_detrended_'+detrending+'_'+model[mm]+version[mm]+'_ensmean_'+obs[oo]+'_'+variables[vv]+'_'+str(years_model[mm][0])+'_'+str(years_model[mm][-1])+'_'+str(season[sea])+'_constantlead.'+figformat
cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, 'aspect':40, 'label':'Hindcast correlation'}
fig = plt.figure()
p = pearson_r.plot(transform=ccrs.PlateCarree(),col='lead',col_wrap=2,subplot_kws={'projection': map_proj},cbar_kwargs=cbar_kwargs)
xx,yy = np.meshgrid(lons,lats)
xx = xx.flatten()
yy = yy.flatten()
for aa in np.arange(len(p.axs.flat)-1):
    ax = p.axs.flat[aa]
    ax.coastlines()
    #plot significant correlation coefficient, optionally taking into account the effective sample size
    if use_effn == 'yes':
        sigbool = pearson_pval_effn[aa,:,:].values.flatten() >= testlevel
    elif use_effn == 'no':
        sigbool = pearson_pval[aa,:,:].values.flatten() >= testlevel
    else:
        raise Exception('ERROR: unknown entry for '+use_effn+'!')
    ax.plot(xx[sigbool],yy[sigbool], linestyle='None', marker='.', markersize=0.5, transform=ccrs.PlateCarree(), markerfacecolor='black', markeredgecolor='black')
    ax.set_extent([lons.min(), lons.max(), lats.min()-south_ext_lambert, lats.max()])
    plt.title(titlelabel)
    #fig.tight_layout()
    plt.savefig(savelabel,dpi=dpival)
    plt.close('all')
#close all nc files and delete associated workspace variable
nc_gcm.close()
nc_obs.close()
del(nc_gcm,nc_obs)
