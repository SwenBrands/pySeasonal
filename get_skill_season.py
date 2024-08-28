#!/usr/bin/env python

'''calculates skill scores for each season and lead over the overlapping time period indicated in <years_model> and <years_obs>.
Also calculates and stores the model quantiles over this period, which are used by pred2tercile.py .'''

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
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#set input parameters
vers = 'v1j_mon' #version number of the output netCDF file to be sent to Predictia
model = ['ecmwf51'] #interval between meridians and parallels
obs = ['era5']
years_model = [1981,2023] #years used in label of model netCDF file, refers to the first and the last year of the monthly model inits
years_obs = [1981,2022] #years used in label of obs netCDF file; if they differ from <years_model>, then the common intersection of years will be validated.
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files

# ##settings for 3-months verification
# season_label = ['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']
# season = [[12,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[11,12,1]] #[[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
# lead = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]] #[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.

##settings for monthly verification
season_label = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
season = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]] #[[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
lead = [[0],[1],[2],[3],[4],[5],[6]] #[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.

# ##settings for monthly verification
# season_label = ['JAN','FEB']
# season = [[1],[2]] #[[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
# lead = [[0],[1]] #[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #number of months between init and start of forecast interval to be verified, e.g. 1 will discard the first month after init, 2 will discard the first two months after init etc.

variables_gcm = ['SPEI-3-R','SPEI-3-M','fwi','msl','t2m','tp','si10','ssrd'] #model variable names in CDS format  GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py> except for <SPEI-3-M> and <SPEI-3-R>, which are paired with <SPEI-3> in <variables_obs>)
variables_obs = ['SPEI-3','SPEI-3','fwi','msl','t2m','tp','si10','ssrd'] #variable names in observations; are identical to <variables_gcm> except for <SPEI-3>, which is referred to as <SPEI-3-M> or <SPEI-3-R> in the model depending on whether past values are taken from the model or reanalysis (i.e. quasi-observations)

# variables_gcm = ['SPEI-3-R','SPEI-3-M'] #model variable names in CDS format  GCM variable names have been set to ERA5 variable names from CDS in <aggregate_hindcast.py> except for <SPEI-3-M> and <SPEI-3-R>, which are paired with <SPEI-3> in <variables_obs>)
# variables_obs = ['SPEI-3','SPEI-3'] #variable names in observations; are identical to <variables_gcm> except for <SPEI-3>, which is referred to as <SPEI-3-M> or <SPEI-3-R> in the model depending on whether past values are taken from the model or reanalysis (i.e. quasi-observations)

datatype = 'float32' #data type of the variables in the output netcdf files
compression_level = 1
precip_threshold = 1/30 #monthly mean daily precipitation threshold in mm below which the modelled and quasi-observed monthly precipitation amount is set to 0; Bring the parameter in pred2tercile.py in exact agreement with this script in future versions
#bin_edges_reliability = np.linspace(0, 1, 4) # probability edges used to calculate the reliability, either <None> in which case the default of xs.reliability() is used (currently 5 bins) or, which should return the same, manually pass 5 bins with np.linspace(0, 1, 6); or 3 bins with np.linspace(0, 1, 4)
bin_edges_reliability = None
corr_outlier = 'no'
aggreg = 'mon' #temporal aggregation of the input files
domain = 'medcof' #spatial domain
int_method = 'conservative_normed' #'conservative_normed', interpolation method used before in <regrid_obs.py>
nr_mem = [25] #considered ensemble members
quantiles = [1/3,1/3*2] #quantiles of the year-to-year time series to be calculated and stored
skillscore_reference = 'rand' #'clim' or 'rand'; use either the climatological mean or randomly reshuffled observed time series as reference forcast for calculating the CPRSS
nr_pseudo_mem = 25 #number of times the climatological mean or randomly reshuffled observarions are replicated along the "member" dimension to mimic a model ensemble representing the naiv climatological forecasts used as reference for calculating the CRPS Skill Score. Results are insensitive to variations in this value; the script has been tested for value=2 and 4.

#options for methods estimating hindcast skill
testlevel = 0.05
detrending = ['yes','no'] #yes or no, linear detrending of the gcm and obs time series prior to validation

#visualization options
figformat = 'png' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
south_ext_lambert = 0 #extend the southern limit of the box containing the Lambert Conformal Projection

## EXECUTE #############################################################
#set basic path structure for observations and gcms
if file_system == 'myLaptop':
    home = os.getenv('HOME')
    rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal'
    path_obs_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/obs/regridded'
    path_gcm_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/gcm/aggregated'
    dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation/'+vers
elif file_system == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    rundir = home+'/Scripts/SBrands/pyPTIclima/pySeasonal'
    path_obs_base = home+'/Results/seasonal/obs/regridded'
    path_gcm_base = home+'/Results/seasonal/gcm/aggregated' 
    dir_netcdf = home+'/Results/seasonal/validation/'+vers
else:
    raise Exception('ERROR: unknown entry for <file_system> input parameter!')
    
#create output directory if it does not exist.
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

#check consistency of some input parameters
if len(season) != len(season_label):
    raise Exception('ERROR: the length of the list <season> does not equal the length of the list <season_label> !')

lead_arr = np.arange(np.array(lead).min(),np.array(lead).max()+1)

#check whether the path to create the output netCDF files exists, create it if not
if os.path.isdir(dir_netcdf) != True:
    os.makedirs(dir_netcdf)

lead_label = [str(lead[ll]).replace(',','').replace('[','').replace(']','').replace(' ','') for ll in np.arange(len(lead))]
#xr.set_options(file_cache_maxsize=1)
for det in np.arange(len(detrending)):
    for vv in np.arange(len(variables_gcm)):
        for mm in np.arange(len(model)):
            if model[mm] == 'ecmwf51' and lead_arr[-1] > 6: #if leadtime > 6 for ecmwf51, the return an error because the 7th forecast month is composed of a few days only in this gcm
                raise Exception('ERROR: the maximum lead requested for '+model[mm]+' ('+str(lead_arr[-1])+') is not valid ! Please check the entries of the input parameter <lead>.')        
            path_gcm = path_gcm_base+'/'+model[mm]+'/'+variables_gcm[vv]+'_'+aggreg+'_'+model[mm]+'_'+str(nr_mem[mm])+'m_'+domain+'_'+str(years_model[0])+'_'+str(years_model[-1])+'.nc'
            nc_gcm = xr.open_dataset(path_gcm)
            #nc_gcm = xr.open_dataset(path_gcm, chunks = {'member' : 1})
            nc_gcm[variables_gcm[vv]] = nc_gcm[variables_gcm[vv]].astype(datatype)
            #leads = np.arange(lead[mm])
            members = nc_gcm.member.values
            nc_gcm = nc_gcm.isel(lead=lead_arr) #filter out the necessary leads only
            
            #set modelled precip. < precip_threshold to 0; note that slightly negative values are present in the monthly precipiation accumulations of ECMWF51
            if variables_gcm[vv] == 'tp':
                print('INFO: setting '+variables_gcm[vv]+' values from '+model[mm]+ '< '+str(precip_threshold)+' to 0...')
                zero_mask = nc_gcm[variables_gcm[vv]].values < precip_threshold
                nc_gcm[variables_gcm[vv]].values[zero_mask] = 0.

            leads = nc_gcm.lead.values
            dates_gcm = pd.DatetimeIndex(nc_gcm.time.values)
            for oo in np.arange(len(obs)):
                print('INFO: validating '+variables_obs[vv]+' from '+model[mm]+' vs '+obs[oo]+' with time series detrending option set to '+detrending[det])
                path_obs = path_obs_base+'/'+obs[oo]+'/'+variables_obs[vv]+'_'+aggreg+'_'+obs[oo]+'_on_'+model[mm]+'_grid_'+int_method+'_'+domain+'_'+str(years_obs[0])+'_'+str(years_obs[-1])+'.nc'
                nc_obs = xr.open_dataset(path_obs)
                nc_obs[variables_obs[vv]] = nc_obs[variables_obs[vv]].astype(datatype)
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
                #check whether the modelled and observed dates are identical
                if np.all(dates_obs.isin(dates_gcm)) == True and np.all(dates_gcm.isin(dates_obs)) == True:
                    print('INFO: the modelled and observed date vectors are identical.')
                    years_common = [dates_gcm.year[0], dates_gcm.year[-1]] #define common start and end year used to save the results in netCDF format
                    print('INFO: the common time period used for verification is '+str(years_common[0])+' to '+str(years_common[1])+' !')
                else:
                    raise Exception('ERROR: the modelled and observed date vectors are not identical !')
                
                #set observed precip. < precip_threshold to 0
                if variables_obs[vv] == 'tp':
                    print('INFO: setting '+variables_obs[vv]+' values from '+obs[oo]+ '< '+str(precip_threshold)+' to 0...')
                    zero_mask = nc_obs[variables_obs[vv]].values < precip_threshold
                    nc_obs[variables_obs[vv]].values[zero_mask] = 0.
               
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
                        
                        #init numpy array which will be filled with tercile values (tercile x season x lead x member x lat x lon)
                        if det == 0 and vv == 0 and mm == 0 and sea == 0 and ll == 0:
                            print('Initializing numpy arrays to be filled with quantiles '+str(quantiles)+'...')
                            quantile_vals = np.zeros((len(detrending),len(variables_gcm),len(model),len(quantiles),nr_seas,nr_leads,nr_mems,nr_lats,nr_lons),dtype='single')
                            quantile_vals[:] = np.nan
                            quantile_vals_ens = np.zeros((len(detrending),len(variables_gcm),len(model),len(quantiles),nr_seas,nr_leads,nr_lats,nr_lons),dtype='single')
                            quantile_vals_ens[:] = np.nan
                        
                        #and loop through each month of this season to calculate the seasonal mean values in both the model and observations
                        for mon in np.arange(len(season[sea])):
                            print('INFO: get values for month '+str(season[sea][mon])+' and leadtime '+str(lead[ll][mon])+'...')
                            monthind_gcm = np.where(np.isin(dates_gcm.month,season[sea][mon]))[0]
                            monthind_obs = np.where(np.isin(dates_obs.month,season[sea][mon]))[0]
                            #check for consistent indices pointing the target month
                            if np.sum(monthind_obs - monthind_gcm) != 0:
                                raise Exception('ERROR: the indices indicating the target month in the observed and model data do not agree !')                    
                            
                            #select the target months/seasons and work with xarray data arrays and numpy arrays from here on instead of working with xr datasets
                            nc_gcm_mon = nc_gcm[variables_gcm[vv]].isel(time=monthind_gcm) #get the requested calendar month from the model
                            nc_gcm_mon = nc_gcm_mon.sel(lead=lead[ll][mon]) #get the requested lead from the model
                            nc_obs_mon = nc_obs[variables_obs[vv]].isel(time=monthind_obs) #get the requested calendar month from the observations / reanalysis, there is no lead time in this case
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
                
                ##generate 6d numpy array with observations replicated along the <member dimension>; will be used as reference for the member-wise GCM verification
                #obs_seas_mn_6d = np.expand_dims(obs_seas_mn_5d,axis=3)
                #obs_seas_mn_6d = np.tile(obs_seas_mn_6d,(1,1,1,len(members),1,1))
               
                #convert numpy array into xarray data arrays
                obs_seas_mn_5d = xr.DataArray(obs_seas_mn_5d,coords=[dates_isea,season_label,lead_label,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'y', 'x'], name=variables_obs[vv]) #convert to xarray data array
                #obs_seas_mn_6d = xr.DataArray(obs_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables_obs[vv])
                gcm_seas_mn_6d = xr.DataArray(gcm_seas_mn_6d,coords=[dates_isea,season_label,lead_label,members,nc_gcm.y,nc_gcm.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables_gcm[vv]) #convert to xarray data array
                gcm_seas_mn_5d = gcm_seas_mn_6d.mean(dim='member') #get ensemble mean values
                
                #optionally apply linear detrending to the time-series, see https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
                if detrending[det] == 'yes':
                    print('INFO: As requested by the user, the gcm and obs time series are linearly detrended.')
                    #gcm_seas_mn_5d_copy = gcm_seas_mn_5d.copy() #was used to check whether the temporal mean of the detrended gcm_seas_mn_5d xr data array is identical to the temporal mean of this non-detrended copy. This answer is yes! 
                    obs_seas_mn_5d = lin_detrend(obs_seas_mn_5d,'no')
                    #obs_seas_mn_6d = lin_detrend(obs_seas_mn_6d,'no')
                    gcm_seas_mn_6d = lin_detrend(gcm_seas_mn_6d,'no')
                    gcm_seas_mn_5d = lin_detrend(gcm_seas_mn_5d,'no')
                elif detrending[det] == 'no':
                    print('INFO: As requested by the user, the gcm and obs time series are not detrended.')
                else:
                    raise Exception('ERROR: check entry for <detrending[det]>')
                
                #calculate the quantiles along the time dimension and put them into the pre-initialized numpy arrays                
                #calculate member-wise quantiles
                quantile_vals_step = gcm_seas_mn_6d.quantile(quantiles, dim='time')
                quantile_vals[det,vv,mm,:,:,:,:,:,:] = quantile_vals_step.values
                
                #shape_gcm = gcm_seas_mn_6d.shape #can be removed !
                
                #get overall / ensemble quantiles calculated upon the array with flattened "member" dimension, thereby mimicing a n member times longer time series
                #gcm_seas_mn_5d_flat_mems = gcm_seas_mn_6d.values.reshape((shape_gcm[0]*shape_gcm[3],shape_gcm[1],shape_gcm[2],shape_gcm[4],shape_gcm[5])) #members are flattened to mimic an extended time dimension, this is a numpy array
                gcm_seas_mn_5d_flat_mems = gcm_seas_mn_6d.transpose('time','member','season','lead','y','x') #change order of the dimensions
                shape_gcm = gcm_seas_mn_5d_flat_mems.shape #get new shape
                gcm_seas_mn_5d_flat_mems = gcm_seas_mn_5d_flat_mems.values.reshape((shape_gcm[0]*shape_gcm[1],shape_gcm[2],shape_gcm[3],shape_gcm[4],shape_gcm[5])) #reshape to concatenate the members along the first / time axis
                quantile_vals_ens_step = np.nanquantile(gcm_seas_mn_5d_flat_mems,quantiles,axis=0)
                quantile_vals_ens[det,vv,mm,:,:,:,:,:] = quantile_vals_ens_step
                
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
                infmask = np.isinf(relbias.values)
                relbias.values[infmask] = np.nan
                del(infmask) #delete to save memory

                #generate 6d numpy array with observations replicated along the <member dimension>; will be used as reference for the member-wise GCM verification
                if skillscore_reference == 'clim': #use climatological mean observations as reference forecast
                    print('As requested by the user, climatological-mean time series are used as reference forecasts')
                    ref_forecast = obs_seas_mn_5d.mean(dim='time').values
                    ref_forecast = np.expand_dims(np.expand_dims(ref_forecast,axis=0),axis=3)
                    ref_forecast = np.tile(ref_forecast,(nr_years,1,1,nr_pseudo_mem,1,1)) #get 6d numpy array containing the naiv climatological forecasts
                    ref_forecast = xr.DataArray(ref_forecast,coords=[dates_isea,season_label,lead_label,members[0:nr_pseudo_mem],nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables_obs[vv]+'_'+skillscore_reference)
                elif skillscore_reference == 'rand': #use randomly reshuffled observed time series as reference forecast
                    print('As requested by the user, randomly reshuffled time series are used as reference forecasts. '+str(nr_pseudo_mem)+' random time series will be generated to this end.')
                    #init the output array of randomly resuffled time series and fill
                    ref_forecast = np.zeros((obs_seas_mn_5d.shape[0],obs_seas_mn_5d.shape[1],obs_seas_mn_5d.shape[2],nr_pseudo_mem,obs_seas_mn_5d.shape[3],obs_seas_mn_5d.shape[4]))
                    ref_forecast[:] = np.nan
                    ntime = len(np.arange(years_common[0],years_common[1]+1)) #number of years
                    for ii in np.arange(nr_pseudo_mem):
                        rand1 = np.random.randint(0,ntime,ntime)
                        ref_forecast[:,:,:,ii,:,:] = obs_seas_mn_5d[rand1,:,:,:,:]
                        del(rand1)
                    ref_forecast = xr.DataArray(ref_forecast,coords=[dates_isea,season_label,lead_label,members[0:nr_pseudo_mem],nc_obs.y,nc_obs.x],dims=['time', 'season', 'lead', 'member', 'y', 'x'], name=variables_obs[vv]+'_'+skillscore_reference)
                else:
                    raise Exception('ERROR: check entry for <skillscore_reference> input parameter !')
                
                #Continuous Ranked Probability Score and respective skill score
                crps_ensemble = xs.crps_ensemble(obs_seas_mn_5d,gcm_seas_mn_6d,member_weights=None,issorted=False,member_dim='member',dim='time',weights=None,keep_attrs=False).rename('crps_ensemble')
                crps_ensemble_clim = xs.crps_ensemble(obs_seas_mn_5d,ref_forecast,member_weights=None,issorted=False,member_dim='member',dim='time',weights=None,keep_attrs=False).rename('crps_ensemble_'+skillscore_reference)
                crps_ensemble_skillscore = 1 - (crps_ensemble/crps_ensemble_clim)
                #close and delete unnecesarry objects to save mem
                crps_ensemble_clim.close()
                del(crps_ensemble_clim)

                # # Reliability
                # #obtain quantile observed quantile thresholds
                # obs_quantile_threshold = obs_seas_mn_5d.quantile(2/3,dim='time') #calculate the quantiles along the time axis
                # obs_quantile_threshold = np.tile(obs_quantile_threshold,(obs_seas_mn_5d.shape[0],1,1,1,1)) #replicate to fit the size of obs_seas_mn_5d; this returns a numpy array
                # #get binary time series, quantile is exceeded yes (1) or no (0)
                # obs_seas_mn_5d_bin = xr.where(obs_seas_mn_5d > obs_quantile_threshold, 1, 0) #here the nan values over the sea are lost and will be put in place again in the next line
                # #obs_seas_mn_5d_bin = xr.where(obs_seas_mn_5d > obs_quantile_threshold, True, False)
                # #obs_seas_mn_5d_bin = obs_seas_mn_5d_bin.where(~np.isnan(obs_seas_mn_5d)) #place nan at their original positions (i.e. over the sea)
                
                # gcm_quantile_threshold = np.tile(quantile_vals_step.sel(quantile=2/3),(obs_seas_mn_5d.shape[0],1,1,1,1,1)) #get upper tercile and add replicate along the newly added time dimension to fit the size of gcm_seas_mn_6d 
                # gcm_seas_mn_6d_bin = xr.where(gcm_seas_mn_6d > gcm_quantile_threshold, 1, 0)
                # #gcm_seas_mn_6d_bin = gcm_seas_mn_6d_bin.where(~np.isnan(gcm_seas_mn_6d)) #place the nan at their original positions
                # o_conditioned_on_y = xs.reliability(obs_seas_mn_5d_bin, gcm_seas_mn_6d_bin.mean("member"), dim='time') #see Wilks 2006, returns the observed relative frequencies (o) conditional to 5 (= default values) forecast probability bins y (0.1, 0.3, 0.5, 0.7, 0.9), see https://xskillscore.readthedocs.io/en/stable/api/xskillscore.reliability.html#xskillscore.reliability 
                # diagonal = np.tile(o_conditioned_on_y.forecast_probability.values,(o_conditioned_on_y.shape[0],o_conditioned_on_y.shape[1],o_conditioned_on_y.shape[2],o_conditioned_on_y.shape[3],1)) #this is the diagonal of the reliability diagramm
                # reliability = np.abs(o_conditioned_on_y - diagonal).mean(dim='forecast_probability') #calculate the residual (i.e. absolute difference) from the diagonal averged over the 5 forecast bins mentioned above
                # reliability = reliability.where(~np.isnan(obs_seas_mn_5d[0,:,:,:,:])) #get grid-boxes over sea to nan as in the input data values, if requested by the user
                
                ## reliability
                reliability_lower = get_reliability_or_roc(obs_seas_mn_5d,gcm_seas_mn_6d,quantile_vals_step,1/3,score_f = 'reliability',bin_edges_f = bin_edges_reliability).rename('reliability_lower_tercile') #calculate reliability for the first tercile
                reliability_upper = get_reliability_or_roc(obs_seas_mn_5d,gcm_seas_mn_6d,quantile_vals_step,2/3,score_f = 'reliability',bin_edges_f = bin_edges_reliability).rename('reliability_upper_tercile') #calculate reliability for the third tercile
                ## roc area under the curve
                roc_auc_lower = get_reliability_or_roc(obs_seas_mn_5d,gcm_seas_mn_6d,quantile_vals_step,1/3,score_f = 'roc_auc').rename('roc_auc_lower_tercile') #calculate roc area under the curve for the first tercile
                roc_auc_upper = get_reliability_or_roc(obs_seas_mn_5d,gcm_seas_mn_6d,quantile_vals_step,2/3,score_f = 'roc_auc').rename('roc_auc_upper_tercile') #calculate roc area under the curve for the third tercile
                
                ## roc area under the curve for climatololgical / reference forecast
                ## explicitly calculate the skill of random forecasts using the randomly re-shuffeled observations generated above (ref_forecast)
                # ref_forecast_quantile_threshold = ref_forecast.quantile(quantiles,dim='time') #get the upper and lower tercile threshold values of the climatological mean observations, these are equal because they are climatological values
                # roc_auc_lower_clim = get_reliability_or_roc(obs_seas_mn_5d,ref_forecast,ref_forecast_quantile_threshold,1/3,score_f = 'roc_auc').rename('roc_auc_lower_tercile_'+skillscore_reference)
                # roc_auc_upper_clim = get_reliability_or_roc(obs_seas_mn_5d,ref_forecast,ref_forecast_quantile_threshold,2/3,score_f = 'roc_auc').rename('roc_auc_upper_tercile_'+skillscore_reference)
                # ref_forecast.close()
                # ref_forecast_quantile_threshold.close()
                # del(ref_forecast,ref_forecast_quantile_threshold)
                # roc_auc_lower_skillscore = (roc_auc_lower - roc_auc_lower_clim) / 1 - roc_auc_lower_clim #roc auc lower tercile skill score
                # roc_auc_upper_skillscore = (roc_auc_upper - roc_auc_upper_clim) / 1 - roc_auc_upper_clim #roc auc upper tercile skill score
                
                quantile_vals_step.close()
                del(quantile_vals_step)
                
                #calc. roc auc skill scores, see https://doi.org/10.1007/s00382-019-04640-4
                roc_auc_lower_skillscore = (roc_auc_lower - 0.5) / 0.5 #roc auc lower tercile skill score
                roc_auc_upper_skillscore = (roc_auc_upper - 0.5) / 0.5 #roc auc upper tercile skill score
                
                #set name of all 3 skill scores
                crps_ensemble_skillscore = crps_ensemble_skillscore.rename('crps_ensemble_skillscore_'+skillscore_reference)
                roc_auc_lower_skillscore = roc_auc_lower_skillscore.rename('roc_auc_lower_tercile_skillscore')
                roc_auc_upper_skillscore = roc_auc_upper_skillscore.rename('roc_auc_upper_tercile_skillscore')
                
                #set infinite values to nan for all 3 skill scores
                infmask_crps = np.isinf(crps_ensemble_skillscore.values)
                crps_ensemble_skillscore.values[infmask_crps] = np.nan
                del(infmask_crps)
                infmask_roc_auc_lower = np.isinf(roc_auc_lower_skillscore.values)
                roc_auc_lower_skillscore.values[infmask_roc_auc_lower] = np.nan
                del(infmask_roc_auc_lower)
                infmask_roc_auc_upper = np.isinf(roc_auc_upper_skillscore.values)
                roc_auc_upper_skillscore.values[infmask_roc_auc_upper] = np.nan
                del(infmask_roc_auc_upper)
                
                # crps_ensemble_skillscore = crps_ensemble_skillscore.where(~np.isinf(crps_ensemble_skillscore))
                # roc_auc_lower_skillscore = roc_auc_lower_skillscore.where(~np.isinf(roc_auc_lower_skillscore))
                # roc_auc_upper_skillscore = roc_auc_upper_skillscore.where(~np.isinf(roc_auc_upper_skillscore))
                
                #add attribures
                pearson_r.attrs['units'] = 'dimensionless'
                pearson_pval.attrs['units'] = 'probability'
                pearson_pval_effn.attrs['units'] = 'probability'
                spearman_r.attrs['units'] = 'dimensionless'
                spearman_pval.attrs['units'] = 'probability'
                spearman_pval_effn.attrs['units'] = 'probability'
                spearman_pval_effn.attrs['units'] = 'probability'
                crps_ensemble.attrs['units'] = 'cummulative probability error'
                crps_ensemble_skillscore.attrs['units'] = 'bound between -1 and 1, positive values indicate more skill than the reference climatological forecast'
                crps_ensemble_skillscore.attrs['reference'] = 'Wilks (2006), pages 281 and 302-304'
                bias.attrs['units'] = nc_obs[variables_obs[vv]].units
                relbias.attrs['units'] = 'percent'
                reliability_lower.attrs['units'] = 'probability'
                reliability_upper.attrs['units'] = 'probability'
                roc_auc_lower.attrs['units'] = 'area under the curve'
                roc_auc_upper.attrs['units'] = 'area under the curve'
                roc_auc_lower_skillscore.attrs['units'] = 'bound between -1 and 1, positive values indicate more skill than the reference climatological forecast'
                roc_auc_upper_skillscore.attrs['units'] = 'bound between -1 and 1, positive values indicate more skill than the reference climatological forecast'
                
                #join xarray dataArrays containing the verification results into a single xarray dataset, set attributes and save to netCDF format
                results = xr.merge((pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn,bias,relbias,crps_ensemble,crps_ensemble_skillscore,reliability_lower,reliability_upper,roc_auc_lower,roc_auc_upper,roc_auc_lower_skillscore,roc_auc_upper_skillscore)) #merge xr dataarrays into a single xr dataset
                del results.attrs['units'] #delete global attributge <units>, which is unexpectedly created by xr.merge() in the previous line; <units> are preserved as variable attribute. 
                #set global and variable attributes
                start_year = str(dates_isea[0])[0:5].replace('-','') #start year considered in the skill assessment
                end_year = str(dates_isea[-1])[0:5].replace('-','') #end year considered in the skill assessment
                results.x['standard_name'] = 'longitude'
                results.y['standard_name'] = 'latitude'
                results.attrs['variable'] = variables_gcm[vv]
                results.attrs['prediction_system'] = model[mm]
                results.attrs['reference_observations'] = obs[oo]
                results.attrs['domain'] = domain
                results.attrs['validation_period'] = start_year+' to '+end_year
                results.attrs['time_series_detrending'] = detrending[det]
                results.attrs['outlier_correction'] = corr_outlier
                results.attrs['version'] = vers
                results.attrs['author'] = 'Swen Brands, brandssf@ifca.unican.es or swen.brands@gmail.com'
                #then save to netCDF and close
                savename_results = dir_netcdf+'/validation_results_season_'+variables_gcm[vv]+'_'+model[mm]+'_vs_'+obs[oo]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending[det]+'_'+start_year+'_'+end_year+'_'+vers+'.nc'
                results.to_netcdf(savename_results)
                
                #retain dimensions used to store quantiles before deleting and closing the respective objects
                if det == len(detrending)-1 and vv == len(variables_gcm)-1 and mm == len(model)-1 and oo == len(obs)-1 and sea == len(season)-1:
                    print('retain dimension attributes for storing the quantiles....')
                    season2quan = results.season
                    lead2quan = results.lead
                    y2quan = results.y.astype(datatype)
                    x2quan = results.x.astype(datatype)
                
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
                crps_ensemble_skillscore.close()
                obs_seas_mn_5d.close()
                #obs_seas_mn_6d.close()
                gcm_seas_mn_5d.close()
                gcm_seas_mn_6d.close()
                reliability_lower.close()
                reliability_upper.close()
                roc_auc_lower.close()
                roc_auc_upper.close()
                #roc_auc_lower_clim.close()
                #roc_auc_upper_clim.close()
                roc_auc_lower_skillscore.close()
                roc_auc_upper_skillscore.close()
                del(results,pearson_r,pearson_pval,pearson_pval_effn,spearman_r,spearman_pval,spearman_pval_effn,bias,relbias,crps_ensemble,crps_ensemble_skillscore,obs_seas_mn_5d,obs_seas_mn_5d_nw,gcm_seas_mn_5d,gcm_seas_mn_6d,gcm_seas_mn_6d_nw,reliability_lower,reliability_upper,roc_auc_lower,roc_auc_upper,roc_auc_lower_skillscore,roc_auc_upper_skillscore)
            
            #close nc files containing observations
            nc_obs.close()
            del(nc_obs)
        #retain attributes and then close and delete nc files containing model data
        x_attrs = nc_gcm.x.attrs
        y_attrs = nc_gcm.x.attrs
        nc_gcm.close()
        del(nc_gcm)

#convert the two numpy arrays containing the two types of quantiles (memberwise and ensemble) into two xarray data arrays, merge them to a single xarray dataset, add attributes to this dataset and save it to netCDF format
quantile_vals = xr.DataArray(quantile_vals, coords=[detrending,variables_gcm,model,np.round(quantiles,2).astype(datatype),season2quan,lead2quan,members,y2quan,x2quan], dims=['detrended','variable','model','quantile_threshold','season','lead','member','y','x'], name='quantile_memberwise')
quantile_vals.attrs['description'] = 'quantile thresholds calculated separately fore each ensemble member' 
quantile_vals_ens = xr.DataArray(quantile_vals_ens, coords=[detrending,variables_gcm,model,np.round(quantiles,2).astype(datatype),season2quan,lead2quan,y2quan,x2quan], dims=['detrended','variable','model','quantile_threshold','season','lead','y','x'], name='quantile_ensemble')
quantile_vals_ens.attrs['description'] = 'quantile thresholds calculated on all ensemble members; the distinct ensemble members have been concatenated along the time dimension prior to calculating the quantiles'
quantile_vals_merged = xr.merge((quantile_vals,quantile_vals_ens))
#dimension attributes
quantile_vals_merged['x'].attrs = x_attrs
quantile_vals_merged['y'].attrs = y_attrs
quantile_vals_merged['detrended'].attrs['info'] = 'Linear de-trending was applied to the modelled and (quasi)observed time series prior to validation; yes or no'
quantile_vals_merged['variable'].attrs['info'] = 'Meteorological variable acronym according to ERA5 nomenclature followed by Copernicus Climate Data Store (CDS)'
quantile_vals_merged['model'].attrs['info'] = 'Name and version of the model / prediction system'
quantile_vals_merged['quantile_threshold'].attrs['info'] = 'Quantile thresholds rounded to 2 decimals'
quantile_vals_merged['season'].attrs['info'] = 'Season the forecast is valid for'
quantile_vals_merged['lead'].attrs['info'] = 'Leadtime of the forecast; one per month'
#global attributes
quantile_vals_merged.attrs['validation_period'] = str(years_common[0])+' to '+str(years_common[1])
quantile_vals_merged.attrs['version'] = vers
quantile_vals_merged.attrs['author'] = "Swen Brands (CSIC-UC, Instituto de Fisica de Cantabria), brandssf@ifca.unican.es or swen.brands@gmail.com"
#save to netCDF
savename_quantiles = dir_netcdf+'/quantiles_pticlima_'+domain+'_'+str(years_common[0])+'_'+str(years_common[1])+'_'+vers+'.nc'
encoding = {'quantile_memberwise': {'zlib': True, 'complevel': compression_level}, 'quantile_ensemble': {'zlib': True, 'complevel': compression_level}}
quantile_vals_merged.to_netcdf(savename_quantiles,encoding=encoding)

#close remaining xarray objects
season2quan.close()
lead2quan.close()
y2quan.close()
x2quan.close()
quantile_vals.close()
quantile_vals_ens.close()
quantile_vals_merged.close()

print('INFO: get_skill_season.py has been run successfully !')
