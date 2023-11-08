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
season_label = ['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDF']
season = [[12,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[11,12,1]] #[[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
lead = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]] #[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.
model = ['ecmwf51'] #interval between meridians and parallels
obs = ['era5']
years_model = [[1981,2022]] #years used in label of model netCDF file, refers to the first and the last year of the monthly model inits
years_obs = [[1981,2022]] #years used in label of obs netCDF file
variables = ['t2m','tp','si10','ssrd'] #variables names valid for both observations and GCM. GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py>
datatype = ['float32','float32','float32','float32']
corr_outlier = 'no'
aggreg = 'mon' #temporal aggregation of the input files
domain = 'medcof' #spatial domain
int_method = 'conservative_normed' #'conservative_normed', interpolation method used in <regrid_obs.py>
nr_mem = [25] #considered ensemble members
nr_pseudo_mem = 2 #number of times the climatological mean observarions are replicated along the "member" dimension to mimic a model ensemble representing the naiv climatological forecasts used as reference for calculating the CRPS Skill Score. Results are insensitive to variations in this value; the script has been tested for value=2 and 4.

#options for methods estimating hindcast skill
testlevel = 0.05
detrending = ['yes','no'] #yes or no, linear detrending of the gcm and obs time series prior to validation

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
#check consistency of some input parameters
if len(season) != len(season_label):
    raise Exception('ERROR: the length of the list <season> does not equal the length of the list <season_label> !')
if len(variables) != len(datatype):
    raise Exception('ERROR: the length of the list <variables> does not equal the length of the list <datatype> !')

lead_arr = np.arange(np.array(lead).min(),np.array(lead).max()+1)

#check whether the path to create the output netCDF files exists, create it if not
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

lead_label = [str(lead[ll]).replace(',','').replace('[','').replace(']','').replace(' ','') for ll in np.arange(len(lead))]
#xr.set_options(file_cache_maxsize=1)
for det in np.arange(len(detrending)):
    for vv in np.arange(len(variables)):
        for mm in np.arange(len(model)):
            if model[mm] == 'ecmwf51' and lead_arr[-1] > 6: #if leadtime > 6 for ecmwf51, the return an error because the 7th forecast month is composed of a few days only in this gcm
                raise Exception('ERROR: the maximum lead requested for '+model[mm]+' ('+str(lead_arr[-1])+') is not valid ! Please check the entries of the input parameter <lead>.')        
            path_gcm = path_gcm_base+'/'+model[mm]+'/'+variables[vv]+'_'+aggreg+'_'+model[mm]+'_'+str(nr_mem[mm])+'m_'+domain+'_'+str(years_model[mm][0])+'_'+str(years_model[mm][-1])+'.nc'
            nc_gcm = xr.open_dataset(path_gcm)
            #nc_gcm = xr.open_dataset(path_gcm, chunks = {'member' : 1})
            nc_gcm[variables[vv]] = nc_gcm[variables[vv]].astype(datatype[vv])
            #leads = np.arange(lead[mm])
            members = nc_gcm.member.values
            nc_gcm = nc_gcm.isel(lead=lead_arr) #filter out the necessary leads only
            
            # #correct potential negative values present in the monthly precipiation accumulations of the GCM, anyway these values are only very slighly negative and thus the correction is not necessary
            # if variables[vv] == 'tp':
                # print('INFO: setting negative values in '+nc_gcm[variables[vv]].name+' from '+model[mm]+' to zero !')
                # neg_mask = nc_gcm[variables[vv]].values < 0
                # nc_gcm[variables[vv]].values[neg_mask] = 0.

            leads = nc_gcm.lead.values
            dates_gcm = pd.DatetimeIndex(nc_gcm.time.values)
            for oo in np.arange(len(obs)):
                print('INFO: validating '+variables[vv]+' from '+model[mm]+' vs '+obs[oo]+' with time series detrending option set to '+detrending[det])
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
                            nr_years = len(dates_isea) #length of the year-to-year time vector for a specific season
                            nr_seas = len(season)
                            nr_leads = len(lead)
                            nr_mems =  len(nc_gcm['member'])                        
                            nr_lats = len(nc_gcm['y'])
                            nr_lons = len(nc_gcm['x'])
                            obs_seas_mn_5d = np.zeros((nr_years,nr_seas,nr_leads,nr_lats,nr_lons))
                            obs_seas_mn_5d[:] = np.nan
                            gcm_seas_mn_6d = np.zeros((nr_years,nr_seas,nr_leads,nr_mems,nr_lats,nr_lons))
                            gcm_seas_mn_6d[:] = np.nan
                            #these inits are for the non weighted (nw) seasonal mean values which will be calculated alongside the weighted values below for comparison
                            obs_seas_mn_5d_nw = np.copy(obs_seas_mn_5d)
                            gcm_seas_mn_6d_nw = np.copy(gcm_seas_mn_6d)
                        
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
                            nc_gcm_mon = nc_gcm_mon.sel(lead=lead[ll][mon]) #get the requested lead from the model
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
                        #calculate weighted seasonal mean values
                        weights = nc_obs_isea.time.dt.days_in_month #get number of days per month for all months of all years forming the time series
                        seasind = np.arange(len(season[sea])) #get index to cut out the n months of the first year, where n is the season length in months, i.e. typically 3
                        weights = weights[seasind] #cut ot the number of days for these months and replicate to match the dimensions of obs_allmon and gcm_allmon
                        weights_obs = np.tile(weights,(nr_years,nr_lats,nr_lons,1))
                        weights_gcm = np.tile(weights,(nr_years,nr_mems,nr_lats,nr_lons,1))
                        obs_seas_mn_5d[:,sea,ll,:,:] = np.sum(obs_allmon * weights_obs, axis = 3) / weights_obs.sum(axis=3)
                        gcm_seas_mn_6d[:,sea,ll,:,:] = np.sum(gcm_allmon * weights_gcm, axis = 4) / weights_gcm.sum(axis=4)
                        #calculate and store unweighted seasonal mean values for comparison
                        obs_seas_mn_5d_nw[:,sea,ll,:,:] = obs_allmon.mean(axis=-1)
                        gcm_seas_mn_6d_nw[:,sea,ll,:,:,:] = gcm_allmon.mean(axis=-1)
                
                #generate 6d numpy array with observations replicated along the <member dimension>; will be used as reference for the member-wise GCM verification
                obs_seas_mn_6d = np.expand_dims(obs_seas_mn_5d,axis=3)
                obs_seas_mn_6d = np.tile(obs_seas_mn_6d,(1,1,1,len(members),1,1))
                
                #convert numpy array into xarray data arrays
                obs_seas_mn_5d = xr.DataArray(obs_seas_mn_5d,coords=[dates_isea,season_label,lead_label,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'y', 'x'], name=variables[vv]) #convert to xarray data array
                obs_seas_mn_6d = xr.DataArray(obs_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv])
                gcm_seas_mn_6d = xr.DataArray(gcm_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_gcm.y,nc_gcm.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv]) #convert to xarray data array
                gcm_seas_mn_5d = gcm_seas_mn_6d.mean(dim='member') #get ensemble meann values
                
                #optionally apply linear detrending to the time-series, see https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
                if detrending[det] == 'yes':
                    print('INFO: As requested by the user, the gcm and obs time series are linearly detrended.')
                    #gcm_seas_mn_5d_copy = gcm_seas_mn_5d.copy() #was used to check whether the temporal mean of the detrended gcm_seas_mn_5d xr data array is identical to the temporal mean of this non-detrended copy. This answer is yes! 
                    obs_seas_mn_5d = lin_detrend(obs_seas_mn_5d,'no')
                    obs_seas_mn_6d = lin_detrend(obs_seas_mn_6d,'no')
                    gcm_seas_mn_6d = lin_detrend(gcm_seas_mn_6d,'no')
                    gcm_seas_mn_5d = lin_detrend(gcm_seas_mn_5d,'no')
                elif detrending[det] == 'no':
                    print('INFO: As requested by the user, the gcm and obs time series are not detrended.')
                else:
                    raise Exception('ERROR: check entry for <detrending[det]>')
                        
                ##start verification
                ##calculalate hindcast correlation coefficient for the inter-annual seasonal-mean time series (observations vs. ensemble mean) and corresponding p-values based on the effective sample size
                #determinstic validation measures
                pearson_r = xs.pearson_r(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_r')
                pearson_pval = xs.pearson_r_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_pval')
                pearson_pval_effn = xs.pearson_r_eff_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('pearson_pval_effn')
                spearman_r = xs.spearman_r(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_r')
                spearman_pval = xs.spearman_r_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_pval')
                spearman_pval_effn = xs.spearman_r_eff_p_value(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('spearman_pval_effn')
                bias = xs.me(obs_seas_mn_5d,gcm_seas_mn_5d,dim='time',skipna=True).rename('bias') #in xskillscore the bias is termed me
                relbias = (bias/obs_seas_mn_5d.mean(dim='time')*100).rename('relbias')
                
                #probabilistic validiation measures
                crps_ensemble = xs.crps_ensemble(obs_seas_mn_5d,gcm_seas_mn_6d,member_weights=None,issorted=False,member_dim='member',dim='time',weights=None,keep_attrs=False).rename('crps_ensemble')
                #get crps for the climatological / no-skill forecast
                obs_clim_mean = obs_seas_mn_5d.mean(dim='time').values
                obs_clim_mean = np.expand_dims(np.expand_dims(obs_clim_mean,axis=0),axis=3)
                #obs_clim_mean = np.tile(obs_clim_mean,(nr_years,1,1,nr_mems,1,1)) #get 6d numpy array containing the naiv climatological forecasts
                obs_clim_mean = np.tile(obs_clim_mean,(nr_years,1,1,nr_pseudo_mem,1,1)) #get 6d numpy array containing the naiv climatological forecasts
                #obs_clim_mean = xr.DataArray(obs_clim_mean,coords=[dates_isea,season_label,lead_label,members,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv]+'_clim') #convert to xarray data array
                obs_clim_mean = xr.DataArray(obs_clim_mean,coords=[dates_isea,season_label,lead_label,members[0:nr_pseudo_mem],nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables[vv]+'_clim')
                crps_ensemble_clim = xs.crps_ensemble(obs_seas_mn_5d,obs_clim_mean,member_weights=None,issorted=False,member_dim='member',dim='time',weights=None,keep_attrs=False).rename('crps_ensemble_clim')
                #close and delete unnecesarry objects to save mem
                obs_clim_mean.close()
                del(obs_clim_mean)
                #calc. skill score
                crps_ensemble_skillscore_clim = 1 - (crps_ensemble/crps_ensemble_clim)
                #close and delete unnecesarry objects to save mem
                crps_ensemble_clim.close()
                del(crps_ensemble_clim)
                #set name of the skill score
                crps_ensemble_skillscore_clim = crps_ensemble_skillscore_clim.rename('crps_ensemble_skillscore_clim')
                
                #add attribures
                pearson_r.attrs['units'] = 'dimensionless'
                pearson_pval.attrs['units'] = 'probability'
                pearson_pval_effn.attrs['units'] = 'probability'
                spearman_r.attrs['units'] = 'dimensionless'
                spearman_pval.attrs['units'] = 'probability'
                spearman_pval_effn.attrs['units'] = 'probability'
                spearman_pval_effn.attrs['units'] = 'probability'
                crps_ensemble.attrs['units'] = 'cummulative probability error'
                bias.attrs['units'] = nc_obs[variables[vv]].units
                relbias.attrs['units'] = 'percent'
                crps_ensemble_skillscore_clim.attrs['units'] = 'bound between -1 and 1, positive values indicate more skill than the reference climatological forecast'
                crps_ensemble_skillscore_clim.attrs['reference'] = 'Wilks (2006), pages 281 and 302-304'
                
                #join xarray dataArrays containing the verification results into a single xarray dataset, set attributes and save to netCDF format
                results = xr.merge((pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn,bias,relbias,crps_ensemble,crps_ensemble_skillscore_clim)) #merge xr dataarrays into a single xr dataset
                del results.attrs['units'] #delete global attributge <units>, which is unexpectedly created by xr.merge() in the previous line; <units> are preserved as variable attribute. 
                #set global and variable attributes
                start_year = str(dates_isea[0])[0:5].replace('-','') #start year considered in the skill assessment
                end_year = str(dates_isea[-1])[0:5].replace('-','') #end year considered in the skill assessment
                results.x['standard_name'] = 'longitude'
                results.y['standard_name'] = 'latitude'
                results.attrs['variable'] = variables[vv]
                results.attrs['prediction_system'] = model[mm]
                results.attrs['reference_observations'] = obs[oo]
                results.attrs['domain'] = domain
                results.attrs['validation_period'] = start_year+' to '+end_year
                results.attrs['time_series_detrending'] = detrending[det]
                results.attrs['outlier_correction'] = corr_outlier
                results.attrs['contact'] = 'Swen Brands, brandssf@ifca.unican.es or swen.brands@gmail.com'
                #then save to netCDF and close
                savename_results = dir_netcdf+'/validation_results_season_'+variables[vv]+'_'+model[mm]+'_vs_'+obs[oo]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending[det]+'_'+start_year+'_'+end_year+'.nc'
                results.to_netcdf(savename_results)
                
                #close and delete remaining temporary xarray objects to free memory
                results.close()
                pearson_r.close()
                pearson_pval.close()
                pearson_pval_effn.close()
                spearman_r.close()
                spearman_pval.close()
                spearman_pval_effn.close()
                bias.close()
                relbias.close()
                crps_ensemble.close()
                crps_ensemble_skillscore_clim.close()
                obs_seas_mn_5d.close()
                obs_seas_mn_6d.close()
                gcm_seas_mn_5d.close()
                gcm_seas_mn_6d.close()
                del(results,pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn,bias,relbias,crps_ensemble,crps_ensemble_skillscore_clim,obs_seas_mn_5d,obs_seas_mn_6d,obs_seas_mn_5d_nw,gcm_seas_mn_5d,gcm_seas_mn_6d,gcm_seas_mn_6d_nw)
                #del(pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn,obs_seas_mn_5d,obs_seas_mn_6d,obs_seas_mn_5d_nw,gcm_seas_mn_5d,gcm_seas_mn_6d,gcm_seas_mn_6d_nw)

            #close nc files containing observations
            nc_obs.close()
        #close nc files containing model data
        nc_gcm.close()
