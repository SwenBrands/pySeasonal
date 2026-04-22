#!/usr/bin/env python
''' This script loads the tercile probabilities for a single model and forecast previously obtained with pred2tercile_operational.py, transforms them into categorical values (1,2,3) indicating the most probable tercile, adds its probability as well as a skill mask; these 3 data variables per variable are saved into a new netCDF file.'''

#load packages
import numpy as np
import xarray as xr
import os
import pdb
import warnings

from pyseasonal.utils.functions_seasonal import (
    apply_sea_mask
)

def swen_seas2ipe(config: dict, year_init: str, month_init: str):
    # # Example year and run to run without passing any input arguments; comment or delete the next two lines in operative use
    # year_init = 2024 #a list containing the years the forecast are initialized on, will be looped through with yy
    # month_init = 10 #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

    # Extract configuration variables
    vers = config['version']
    model = config['model']
    version = config['model_version']
    years_quantile = config['years_quantile']
    subperiod = config['subperiod']
    scores = config['scores'] #pointing to the names of the binary skill masks for ROC-AUC skill scores of each tercile
    agg_labels = config['agg_labels']
    nan_placeholder = config['nan_placeholder']

    variables_std = config['variables_std']
    variables_out = config['variables_out']

    datatype = config['datatype']
    domain = config['domain']
    detrended = config['detrended']
    quantile_threshold = config['quantile_threshold']

    # Get paths from configuration
    paths = config['paths']

    ## EXECUTE #############################################################

    # Check consistency of input parameters
    if len(variables_std) != len(variables_out):
        raise ValueError('<variables_std> and <variables_out> must have the same length !')

    # Extract paths from configuration
    home = paths['home']
    server = paths['server']
    path_gcm_base = paths['path_gcm_base']
    path_gcm_base_derived = paths['path_gcm_base_derived']
    path_gcm_base_masked = paths['path_gcm_base_masked']
    dir_validation = paths['dir_validation'] + '/' + vers
    dir_forecast = paths['dir_forecast']
    dir_output = paths['dir_output']
    mask_dir = paths['mask_dir']

    print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

    #create output directory of the forecasts generated here, if it does not exist.
    if os.path.isdir(dir_output) != True:
        os.makedirs(dir_output)
    
    #get path to mask file as a function of the requested domain
    if domain == 'medcof':
        mask_file_indir = 'ECMWF_Land_Medcof_descending_lat_reformatted.nc' # mask file as it appears in its directory
    elif domain == 'Iberia':
        mask_file_indir = 'PTI-grid_Iberia_010_descending_lat_reformatted.nc'
    elif domain == 'Canarias':
        # mask_file_indir = 'PTI-grid_Canarias_descending_lat_reformatted.nc'
        mask_file_indir = 'PTI-grid_Canarias_0025_descending_lat_reformatted.nc'
    else:
        raise ValueError('Check entry for <domain> input parameter !')
    mask_file = mask_dir+'/'+mask_file_indir #here, descending lats are needed (check why the DataArrays behave distinct concerning ascending or descending lats in pySeasonal)

    
    #loop through aggregation windows, models, and variables
    for ag in np.arange(len(agg_labels)):
        for mm in np.arange(len(model)):

            nc_forecast = xr.Dataset() #create empty xarray dataset to be filled with xr data arrays in the next loop
            for vv in np.arange(len(variables_std)):

                #load the skill masks for a given aggregation window; multiple variables are and models located within the same file.
                filename_validation = dir_validation+'/skill_masks/'+domain+'/'+agg_labels[ag]+'/skill_masks_pticlima_'+domain+'_'+agg_labels[ag]+'_'+model[mm]+version[mm]+'_'+variables_std[vv]+'_'+vers+'.nc'
                nc_val = xr.open_dataset(filename_validation)

                #check if model name in skill mask file matches the requested model name
                if nc_val.model != model[mm]+version[mm]:
                    raise ValueError('The model name requested by <model[mm]> does not match the model name stored in <nc_val> !')

                #load the forecast for a specific variable
                filename_forecast = dir_forecast+'/'+domain+'/probability_'+agg_labels[ag]+'_'+model[mm]+version[mm]+'_'+variables_std[vv]+'_'+domain+'_init_'+str(year_init)+str(month_init).zfill(2)+'_dtr_'+detrended+'_refyears_'+str(years_quantile[mm][0])+'_'+str(years_quantile[mm][1])+'_'+vers+'.nc'
                nc_forecast_in = xr.open_dataset(filename_forecast) #get the xr data array containing the tercile probabilities for a specific variable

                #check whether the previously stored model and version thereof match those requested by this script
                if nc_forecast_in.model == model[mm]+version[mm]:
                    print('The requested model '+model[mm]+' and its version '+version[mm]+' coincide with the entry previously stored in '+filename_forecast+' !')
                else:
                    raise ValueError('The requested model '+model[mm]+' and its version '+version[mm]+' do NOT coincide with the entry previously stored in '+filename_forecast+' !')

                tercile_attrs = nc_forecast_in.tercile.attrs

                nc_forecast_in_prob = nc_forecast_in['probability'].sel(aggregation=agg_labels[ag],model=model[mm]+version[mm]).drop_vars(['aggregation','model'])

                #get maximum probability, #note that nc_forecast_in_prob only contains the probability of the most likely tercile ! The other two terciles are set to nan by pred2tercile_operational.py
                nc_forecast_in_maxprob = nc_forecast_in_prob.max(dim='tercile').astype(datatype)
                valid_data_bool_4d = ~np.isnan(nc_forecast_in_maxprob) #get valid data for the 4d array
                valid_data_bool_5d = ~np.isnan(nc_forecast_in_prob) #get valid data for the 5d array
                nc_forecast_in_argmax = nc_forecast_in_prob.where(valid_data_bool_5d, other=nan_placeholder).argmax(dim='tercile')+1 #set nan values to 0 and calculate tercile position of the maxiumum probability value (1,2 or 3)
                nc_forecast_in_argmax = nc_forecast_in_argmax.where(valid_data_bool_4d,other=np.nan) #set tercile position to nan where maxima are nan

                #clean
                nc_forecast_in_prob.close()
                del(nc_forecast_in_prob)

                nc_forecast['mlt_'+variables_out[vv]] = nc_forecast_in_argmax
                nc_forecast['prob_'+variables_out[vv]] = nc_forecast_in_maxprob
                #add variable attributes
                nc_forecast['mlt_'+variables_out[vv]].attrs['long_name'] = 'Most likely tercile category (1=lower, 2=middle, 3=upper) for '+variables_out[vv]
                nc_forecast['mlt_'+variables_out[vv]].attrs['standard_name'] = 'tercile_category'
                nc_forecast['mlt_'+variables_out[vv]].attrs['units'] = '1'
                try:
                    nc_forecast['mlt_'+variables_out[vv]].attrs[variables_out[vv]+'_description'] = nc_forecast_in.description
                except:
                    warnings.warn('No variable or index description available for '+variables_out[vv])

                nc_forecast['prob_'+variables_out[vv]].attrs['long_name'] = 'probability_of_most_likely_tercile_for_'+variables_out[vv]
                nc_forecast['prob_'+variables_out[vv]].attrs['standard_name'] = 'probability_of_event_in_category'
                nc_forecast['prob_'+variables_out[vv]].attrs['units'] = '1'
                try:
                    nc_forecast['prob_'+variables_out[vv]].attrs[variables_out[vv]+'_description'] = nc_forecast_in.description
                except:
                    warnings.warn('No variable or index description available for '+variables_out[vv])

                #filter subperiod, detrending option, model and version, season, and variable from skill mask
                fc_season_length = nc_forecast.season.length_in_months

                #check whether the requested aggregation window matches the season length stored within the file
                if fc_season_length != int(agg_labels[ag][0]):
                    raise ValueError('<fc_season_length> must equal <agg_labels[ag]> ! ')

                fc_seasons = nc_forecast.season.values
                #get the binary skill mask for a given subperiod, detrending option and season
                nc_val_sub = nc_val.sel(subperiod=subperiod,detrended=detrended,season=fc_seasons)[scores]
                val_leads = nc_val_sub.lead.values

                skill_mask = np.zeros((len(scores),len(fc_seasons),len(nc_val.y),len(nc_val.x))) #a 3d array
                for sc in np.arange(len(scores)):
                    for sea in np.arange(len(fc_seasons)):
                        skill_mask_step = nc_val_sub[scores[sc]].sel(season=fc_seasons[sea],lead=val_leads[sea])
                        skill_mask[sc,sea,:,:] = skill_mask_step.values

                #bring numpy array into xarray data array format
                skill_mask = xr.DataArray(skill_mask.astype('float32'),coords=[scores,fc_seasons,nc_val.y,nc_val.x],dims=['scores','season','y','x'], name=variables_std[vv]+'_skill_mask')

                #check whether the x and y coordinates in the forecast and mask files are identical
                if np.all(nc_forecast.y == nc_val.y) and np.all(nc_forecast.x == nc_val.x):
                    print('The x and y values in <nc_forecast> and <nc_val> are identical ! Proceed to find the skill mask entry for each tercile and season....')
                else:
                    raise ValueError('The x and y values in <nc_forecast> and <nc_val> are NOT identical !')
                
                ## cut down the forecast domain to the skill domain indicated in the <domain> input variable
                # nc_forecast = nc_forecast.sel(y=skill_mask.y,x=skill_mask.x)

                #select the binary skill mask value of the most likely tercile; since the most likely tercile starts with 1, a 1 is subtracted to the index array prior to filtering out the binary skill values, nan are set to 0 and must be placed again to mark values over the sea
                # # either (exact):
                # mlt_index = nc_forecast.squeeze()['mlt_'+variables_out[vv]] #get like tercile in the forecasat (nan, 1, 2 or 3)
                # nan_locs = np.where(np.isnan(mlt_index.values)) # get indices of the nan values
                # skill_mask = skill_mask.isel(scores = mlt_index.fillna(1).astype(int)-1) # set nan values to 1 and subtract 1 to get index values (0, 1, or 2); take this index values to select the binary skill value for the most likely tercile in the forecast
                # skill_mask[nan_locs[0]] = np.nan #place the original nan values again

                # or (simpler but not exact since nan values can existe over land that would be set to 1):
                skill_mask = skill_mask.isel(scores=nc_forecast.squeeze()['mlt_'+variables_out[vv]].fillna(1).astype(int)-1).drop_vars('scores') # note that the <rtime> coordinate is passed from <nc_forecast> to <skill_mask> in this line 
                
                # search the binary skill value for the specific forecasted tercile on each grid-box

                # apply land-sea mask to skill mask
                skill_mask = apply_sea_mask(skill_mask, mask_file, 'y', 'x')

                # add attributes to skill mask
                scores_label = str(scores).replace("[","").replace("]","").replace("'","")
                skill_mask.attrs['long_name'] = 'binary_skill_mask_based_on_'+scores_label+' for '+variables_out[vv]
                skill_mask.attrs['standard_name'] = 'binary_skill_mask'
                skill_mask.attrs['units'] = 'binary'
                try:
                    skill_mask.attrs['description'] = nc_forecast_in.description
                except:
                    warnings.warn('No variable or index description available for '+variables_out[vv])

                # merge skill mask into the existing xarray dataset containing the most like tercile and its probability
                nc_forecast['skill_'+variables_out[vv]] = skill_mask #assign variable-specific skill mask to the newly generated xarray dataset produced by this script
                
                #clean
                nc_val.close(), nc_val_sub.close(); nc_forecast_in_argmax.close(); nc_forecast_in_maxprob.close(); nc_forecast_in.close(); skill_mask.close()
                del(nc_val, nc_val_sub, nc_forecast_in_argmax,nc_forecast_in_maxprob,nc_forecast_in,skill_mask)

            #add global attributes
            nc_forecast.season.attrs['standard_name'] = 'season'
            nc_forecast.season.attrs['long_name'] = 'season following the forecast initialization date specified in rtime'

            # save the mergerd output file
            savename = dir_output+'/seas2ipe_'+model[mm]+version[mm]+'_'+domain+'_init_'+str(year_init)+str(month_init).zfill(2)+'_'+str(fc_season_length)+'mon_dtr_'+detrended+'_refyears_'+str(tercile_attrs['tercile_period'][0])+'_'+str(tercile_attrs['tercile_period'][1])+'.nc'

            ## optionally encode the output netcdf file
            # chunks = (1, 1, 1, 1, len(nc_forecast.y), len(nc_forecast.y))
            # encoding = dict(tp=dict(chunksizes=chunks)) #https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
            # nc_forecast.to_netcdf(savename,encoding=encoding)
            
            nc_forecast.to_netcdf(savename)

            #clean up
            nc_forecast.close()
            del(nc_forecast)