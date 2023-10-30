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
model = ['ecmwf51'] #interval between meridians and parallels
obs = ['era5']
years_model = [[1981,2022]] #years used in label of model netCDF file, refers to the first and the last year of the monthly model inits
years_obs = [[1981,2022]] #years used in label of obs netCDF file
variables = ['tp'] #['tp','t2m'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>
corr_outlier = 'no'
datatype = ['float32'] #['float32','float32']
aggreg = 'mon' #temporal aggregation of the input files
domain = 'medcof' #spatial domain
int_method = 'conservative_normed' #'conservative_normed', interpolation method used in <regrid_obs.py>
nr_mem = [25] #considered ensemble members

#options for methods estimating hindcast skill
testlevel = 0.05
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

## EXECUTE #############################################################
#check whether the path to create the output netCDF files exists, create it if not
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

lead_label = [str(lead[ll]).replace(',','').replace('[','').replace(']','').replace(' ','') for ll in np.arange(len(lead))]
#xr.set_options(file_cache_maxsize=1)
for vv in np.arange(len(variables)):
    print('INFO: validating '+variables[vv])
    for mm in np.arange(len(model)):
        path_gcm = path_gcm_base+'/'+model[mm]+'/'+variables[vv]+'_'+aggreg+'_'+model[mm]+'_'+str(nr_mem[mm])+'m_'+domain+'_'+str(years_model[mm][0])+'_'+str(years_model[mm][-1])+'.nc'
        nc_gcm = xr.open_dataset(path_gcm)
        #nc_gcm = xr.open_dataset(path_gcm, chunks = {'member' : 1})
        nc_gcm[variables[vv]] = nc_gcm[variables[vv]].astype(datatype[vv])
        #leads = np.arange(lead[mm])
        members = nc_gcm.member.values
        nc_gcm = nc_gcm.isel(lead=np.arange(np.array(lead).min(),np.array(lead).max()+1)) #filter out the necessary leads only
        leads = nc_gcm.lead.values
        dates_gcm = pd.DatetimeIndex(nc_gcm.time.values)
        for oo in np.arange(len(obs)):
            path_obs = path_obs_base+'/'+obs[oo]+'/'+variables[vv]+'_'+aggreg+'_'+obs[oo]+'_on_'+model[mm]+'_grid_'+int_method+'_'+domain+'_'+str(years_obs[oo][0])+'_'+str(years_obs[oo][-1])+'.nc'
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
            #redefined date vectors for common time period
            dates_obs = pd.DatetimeIndex(nc_obs.time.values)
            dates_gcm = pd.DatetimeIndex(nc_gcm.time.values)
           
            #loop through each season
            for sea in np.arange(len(season)):                    
                #loop through list of monthly lead times
                for ll in np.arange(len(lead)):
                    #get dates for the inter-annual time series containing the requested seasons
                    #seasind_obs = nc_obs.time.dt.season == season_label[sea] #get time indices pointing to all months forming the requested season
                    seasind_obs = np.where(np.isin(dates_obs.month,season[sea]))[0] #get time indices pointing to all months forming the requested season
                    nc_obs_isea = nc_obs.isel(time = seasind_obs)
                    dates_isea = pd.DatetimeIndex(nc_obs_isea.time.values)
                    #retain only one date entry per year to pair with seaonal mean averages calculated below
                    seasind_obs_for_dates = np.where(dates_isea.month == season[sea][0])[0]
                    dates_isea = dates_isea[seasind_obs_for_dates]
                    #nc_obs_isea_for_dates = nc_obs_isea.isel(time = seasind_obs_for_dates)
                    #dates_isea = pd.DatetimeIndex(nc_obs_isea_for_dates.time.values)
                    
                    #init numpy array which will be filled with seasonal mean values (time x season x lead x member x lat x lon)
                    if sea == 0 and ll == 0:
                        #obs_seas_mn_5d = np.zeros((len(dates_isea,len(season),len(lead),len(nc_obs_isea.y),len(nc_obs_isea.x))
                        obs_seas_mn_5d = np.zeros((len(dates_isea),len(season),len(lead),len(nc_obs_isea.y),len(nc_obs_isea.x)))
                        obs_seas_mn_5d[:] = np.nan
                        gcm_seas_mn_6d = np.zeros((len(dates_isea),len(season),len(lead),len(members),len(nc_obs_isea.y),len(nc_obs_isea.x)))
                        gcm_seas_mn_6d[:] = np.nan
                        #gcm_seas_mn_ensmean_5d = np.zeros((len(dates_isea,len(season),len(lead),len(nc_obs_isea.y),len(nc_obs_isea.x))
                    
                    #and loop through each month of this season to calculate the seasonal mean values in both the model and observations
                    for mon in np.arange(len(season[sea])):
                        print('INFO: get values for month '+str(season[sea][mon])+' and leadtime '+str(lead[ll][mon])+'...')
                        monthind_gcm = np.where(np.isin(dates_gcm.month,season[sea][mon]))[0]
                        monthind_obs = np.where(np.isin(dates_obs.month,season[sea][mon]))[0]
                        #check for consistent indices pointing the target month
                        if np.sum(monthind_obs - monthind_gcm) != 0:
                            raise Exception('ERROR: the indices indicating the target month in the observed and model data do not agree !')                    
                        
                        #select the target months/seasons and work with xarray data arrays and numpy arrays from here on instead of working with xr datasets
                        nc_gcm_mon = nc_gcm[variables[vv]].isel(time=monthind_gcm) #get the requested calendar month from the model
                        #nc_gcm_mon = nc_gcm_mon.sel(lead=lead[sea][mon]) #get the requested lead from the model
                        nc_gcm_mon = nc_gcm_mon.sel(lead=lead[sea][ll]) #get the requested lead from the model
                        nc_obs_mon = nc_obs[variables[vv]].isel(time=monthind_obs) #get the requested calendar month from the observations / reanalysis, there is no lead time in this case
                        gcm_mon = nc_gcm_mon.values #retain the numpy arrays with in the xr dataarrays
                        obs_mon = nc_obs_mon.values #retain the numpy arrays with in the xr dataarrays
                        
                        #set values to nan for first January and Feburary values in the time series forming the DJF-mean value. The J and F values have no D pair at the start of fhe time series.
                        if (season[sea] == [12,1,2] and str(season[sea][mon]) in ('1','2')) or (season[sea] == [11,12,1] and str(season[sea][mon]) in ('1')):
                            print('INFO: target season is '+str(season[sea])+', target month is '+str(season[sea][mon])+'. The first value is thus set to NaN!')
                            gcm_mon[0,:,:,:] = np.nan
                            obs_mon[0,:,:] = np.nan
                        # elif season[sea] == [11,12,1] and str(season[sea][mon]) in ('1','2'):
                            # print('INFO: target season is '+str(season[sea])+', target month is '+str(season[sea][mon])+'. The first value is thus set to NaN!')
                            # gcm_mon[0,:,:,:,:] = np.nan
                            # obs_mon[0,:,:,:] = np.nan
                        else:
                            print('INFO: target season is '+str(season[sea])+', target month is '+str(season[sea][mon]))

                        ##init numpy array with additional dimension <month> whose lenght equals the number of months in season[sea]
                        if mon == 0:
                            gcm_allmon = np.zeros((gcm_mon.shape[0],gcm_mon.shape[1],gcm_mon.shape[2],gcm_mon.shape[3],len(season[sea])))
                            gcm_allmon[:] = np.nan
                            obs_allmon = np.zeros((obs_mon.shape[0],obs_mon.shape[1],obs_mon.shape[2],len(season[sea])))
                            obs_allmon[:] = np.nan
                        #close temporary xarray objects
                        nc_gcm_mon.close()
                        nc_obs_mon.close()
                        nc_obs_isea.close()
                        
                        #fill the initialized arrays
                        gcm_allmon[:,:,:,:,mon] = gcm_mon
                        obs_allmon[:,:,:,mon] = obs_mon
                    #calculate seasonal mean values, calculate weighted mean in future versions
                    obs_seas_mn_5d[:,sea,ll,:,:] = obs_allmon.mean(axis=-1)
                    gcm_seas_mn_6d[:,sea,ll,:,:,:] = gcm_allmon.mean(axis=-1)
                    #gcm_seas_mn_ensmean_5d = np.mean(gcm_seas_mn,axis=3) #caculate numpy array with ensemble mean values
            
            #generate 6d numpy array with observations replicated along the <member dimension>; will be used as reference for the member-wise GCM verification
            obs_seas_mn_6d = np.expand_dims(obs_seas_mn_5d,axis=3)
            obs_seas_mn_6d = np.tile(obs_seas_mn_6d,(1,1,1,len(members),1,1))
            
            #convert numpy array into xarray data arrays
            obs_seas_mn_5d = xr.DataArray(obs_seas_mn_5d,coords=[dates_isea,season_label,lead_label,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'y', 'x'], name=variables[vv]) #convert to xarray data array
            obs_seas_mn_6d = xr.DataArray(obs_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv])
            gcm_seas_mn_6d = xr.DataArray(gcm_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_gcm.y,nc_gcm.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv]) #convert to xarray data array
            gcm_seas_mn_5d = gcm_seas_mn_6d.mean(dim='member') #get ensemble meann values
            
            #optionally apply linear detrending to the time-series, see https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
            if detrending == 'yes':
                print('INFO: As requested by the user, the gcm and obs time series are linearly detrended along the time dimension.')
                obs_seas_mn_5d = lin_detrend(obs_seas_mn_5d)
                obs_seas_mn_6d = lin_detrend(obs_seas_mn_6d)
                gcm_seas_mn_6d = lin_detrend(gcm_seas_mn_6d)
                gcm_seas_mn_5d = lin_detrend(gcm_seas_mn_5d)
            elif detrending == 'no':
                print('INFO: As requested by the user, the gcm and obs time series are not detrended along the time dimension.')
            else:
                raise Exception('ERROR: check entry for <detrending>')
                    
            ##start verification
            ##calculalate hindcast correlation coefficient for the inter-annual seasonal-mean time series (observations vs. ensemble mean) and corresponding p-values based on the effective sample size
            pearson_r = xs.pearson_r(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_r')
            pearson_pval = xs.pearson_r_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_pval')
            pearson_pval_effn = xs.pearson_r_eff_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_pval_effn')
            spearman_r = xs.spearman_r(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_r')
            spearman_pval = xs.spearman_r_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_pval')
            spearman_pval_effn = xs.spearman_r_eff_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_pval_effn')
            
            #add attribures
            pearson_r.attrs['units'] = 'dimensionless'
            pearson_pval.attrs['units'] = 'probability'
            pearson_pval_effn.attrs['units'] = 'probability'
            spearman_r.attrs['units'] = 'dimensionless'
            spearman_pval.attrs['units'] = 'probability'
            spearman_pval_effn.attrs['units'] = 'probability'
            
            #join xarray dataArrays containing the verification results into a single xarray dataset, set attributes and save to netCDF format
            results = xr.merge((pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn)) #merge xr dataarrays into a single xr dataset
            del results.attrs['units'] #delete global attributge <units>, which is unexpectedly created by xr.merge() in the previous line; <units> are preserved as variable attribute. 
            #set global and variable attributes
            start_year = str(dates_isea[0])[0:5].replace('-','') #start year considered in the skill assessment
            end_year = str(dates_isea[-1])[0:5].replace('-','') #end year considered in the skill assessment
            results.x['standard_name'] = 'longitude'
            results.y['standard_name'] = 'latitude'
            results.attrs['prediction_system'] = model[mm]
            results.attrs['reference_observations'] = obs[oo]
            results.attrs['validation_period'] = start_year+' to '+end_year
            results.attrs['time_series_detrending'] = detrending
            results.attrs['outlier_correction'] = corr_outlier
            results.attrs['contact'] = 'Swen Brands, brandssf@ifca.unican.es or swen.brands@gmail.com'
            #then save to netCDF and close
            savename_results = dir_netcdf+'/verification_results_season_'+variables[vv]+'_'+model[mm]+'_vs_'+obs[oo]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending+'_'+start_year+'_'+end_year+'.nc'
            results.to_netcdf(savename_results)
            results.close()
            
        #close nc files containing observations
        nc_obs.close()
    #close nc files containing model data
    nc_gcm.close()
