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
#season = ['DJF']
season = [1,2,3,4,5,6,7,8,9,10,11,12]
n_lead = [7] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.
model = ['ecmwf'] #interval between meridians and parallels
version = ['51']
obs = ['era5']
years_model = [[1981,2022]] #years used in label of model netCDF file, refers to the first and the last year of the monthly model inits
years_obs = [[1981,2022]] #years used in label of obs netCDF file
variables = ['tp','t2m'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>
datatype = ['float32','float32']
aggreg = 'mon' #temporal aggregation of the input files
domain = 'medcof' #spatial domain
int_method = 'conservative_normed' #'conservative_normed', interpolation method used in <regrid_obs.py>
nr_mem = [25] #considered ensemble members
window = 3 #number of consecutive months considered for calculating the rolling seasonal average values

#options for methods estimating hindcast skill
testlevel = 0.05
use_effn = 'yes' #use effective sample size to calculate the p-value of the Pearson and Spearman correlation coefficient.
detrending = 'no' #yes or no, linear detrending of the gcm and obs time series prior to validation

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

#set basic path structure for observations and gcms
home = os.getenv('HOME')
path_obs_base = home+'/datos/tareas/pticlima/seasonal/results/obs/regridded'
path_gcm_base = home+'/datos/tareas/pticlima/seasonal/results/gcm/aggregated'
path_figures = home+'/datos/tareas/pticlima/seasonal/figures'

## EXECUTE #############################################################
#xr.set_options(file_cache_maxsize=1)
for vv in np.arange(len(variables)):
    for mm in np.arange(len(model)):
        path_gcm = path_gcm_base+'/'+model[mm]+version[mm]+'/'+variables[vv]+'_'+aggreg+'_'+model[mm]+version[mm]+'_'+str(nr_mem[mm])+'m_'+domain+'_'+str(years_model[mm][0])+'_'+str(years_model[mm][-1])+'.nc'
        #nc_gcm = xr.open_dataset(path_gcm)
        nc_gcm = xr.open_dataset(path_gcm, chunks = {'member' : 1})
        nc_gcm[variables[vv]] = nc_gcm[variables[vv]].astype(datatype[vv])
        leads = np.arange(n_lead[mm])
        members = nc_gcm.member.values
        nc_gcm = nc_gcm.isel(lead=leads)
        dates_gcm = pd.DatetimeIndex(nc_gcm.time.values)
        for oo in np.arange(len(obs)):
            path_obs = path_obs_base+'/'+obs[oo]+'/'+variables[vv]+'_'+aggreg+'_'+obs[oo]+'_on_'+model[mm]+version[mm]+'_grid_'+int_method+'_'+domain+'_'+str(years_obs[oo][0])+'_'+str(years_obs[oo][-1])+'.nc'
            nc_obs = xr.open_dataset(path_obs)
            nc_obs[variables[vv]] = nc_obs[variables[vv]].astype(datatype[vv])
            dates_obs = pd.DatetimeIndex(nc_obs.time.values)
            dates_bool_obs = dates_obs.isin(dates_gcm)
            dates_bool_gcm = dates_gcm.isin(dates_obs)
            dates_obs = dates_obs[dates_bool_obs]
            dates_gcm = dates_gcm[dates_bool_gcm]
            
            #select common time period
            nc_gcm = nc_gcm.isel(time = dates_bool_gcm)
            nc_obs = nc_obs.isel(time = dates_bool_obs)
            
            #calculate rolling seasonal-mean values with constant lead-time (i.e. 1-month lead time for all months contained in a given season DJF, MAM, JJA or SON
            if all(isinstance(item, str) for item in season): #if all instances of the the list <season> are string, then seasonal evaluation is assumed and seasonal averages are calculated
                print('INFO: seasonal averages are calcualted in preparation of the seasonal skill evaluation requested by the user...')
                nc_obs = calc_roll_seasmean(nc_obs)
                nc_gcm = calc_roll_seasmean(nc_gcm)
            elif all(isinstance(item, int) for item in season):
                print('INFO: seasonal averages are not calculated since the skill evulation requested by the user is monthly...')
            else:
                raise Exception('ERROR: unknown entry in input parameter <season> !')
           
            for sea in np.arange(len(season)):
                if type(season[sea]) == str:
                    print('INFO: as requested by the user in <season> = '+str(season[sea])+', a seasonal skill evaluation with fixed lead-time per individual month is performed...')
                    obs_seasind = nc_obs.time.dt.season == season[sea]
                    gcm_seasind = nc_gcm.time.dt.season == season[sea]
                elif type(season[sea]) == int:
                    print('INFO: as requested by the user in <season> = '+str(season[sea])+', a monthly skill evaluation is performed...')
                    obs_seasind = nc_obs.time.dt.month == season[sea]
                    gcm_seasind = nc_gcm.time.dt.month == season[sea]
                else:
                    raise Exception('ERROR: unknown entry in <season>!')
                    
                nc_obs_isea = nc_obs.isel(time = obs_seasind)
                nc_gcm_isea = nc_gcm.isel(time = gcm_seasind)
                dates_isea = pd.DatetimeIndex(nc_obs_isea.time.values)

                #calculate ensemble mean values
                nc_gcm_isea_mn = nc_gcm_isea.mean(dim='member')
                #replicate observations x1 times, where x1 is the leadtime in months defined in <lead> above and then, additionally, x2 times, where x2 is the number of ensemble members. This is to have the same dimensions in the obs and (optionally ensemble mean) gcm data 
                obs_4d = np.expand_dims(nc_obs_isea[variables[vv]].values,1).repeat(nc_gcm_isea[variables[vv]].shape[1],axis=1) #obs_4d is a numpy array               
                obs_5d = np.expand_dims(obs_4d,2).repeat(nc_gcm_isea[variables[vv]].shape[2],axis=2) #obs_5d is a numpy array as well
                
                #convert to xarray DataArray for further processing
                obs_4d = xr.DataArray(obs_4d,coords=[dates_isea,leads,nc_obs_isea.y,nc_obs_isea.x],dims=['time', 'lead', 'y', 'x'], name=variables[vv])
                obs_5d = xr.DataArray(obs_5d,coords=[dates_isea,leads,members,nc_obs_isea.y,nc_obs_isea.x],dims=['time', 'lead', 'member', 'y', 'x'], name=variables[vv])
                
                #optionally apply linear detrending to the time-series, see https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
                if detrending == 'yes':
                    print('INFO: As requested by the user, the gcm and obs time series are linearly detrended along the time dimension.')
                    nc_gcm_isea = lin_detrend(nc_gcm_isea[variables[vv]])
                    obs_4d = lin_detrend(obs_4d)
                elif detrending == 'no':
                    print('INFO: As requested by the user, the gcm and obs time series are not detrended along the time dimension.')
                else:
                    raise Exception('ERROR: check entry for <detrending>')
                
                ##start verification
                ##calculalate hindcast correlation coefficient for the inter-annual seasonal-mean time series (observations vs. ensemble mean) and corresponding p-values based on the effective sample size
                pearson_r = xs.pearson_r(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('pearson_r')
                pearson_pval = xs.pearson_r_p_value(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('pearson_pval')
                pearson_pval_effn = xs.pearson_r_eff_p_value(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('pearson_pval_effn')
                spearman_r = xs.spearman_r(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('spearman_r')
                spearman_pval = xs.spearman_r_p_value(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('spearman_pval')
                spearman_pval_effn = xs.spearman_r_eff_p_value(obs_4d,nc_gcm_isea_mn[variables[vv]],dim='time',skipna=True).rename('spearman_pval_effn')
               
                #map results
                lons = pearson_r.x.values
                lats = pearson_r.y.values
                # This is the map projection we want to plot *onto*, see https://docs.xarray.dev/en/stable/examples/visualization_gallery.html
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
        #close all nc files and delete associated workspace varialbe
        nc_gcm.close()
        nc_obs.close()
        del(nc_gcm,nc_obs)
