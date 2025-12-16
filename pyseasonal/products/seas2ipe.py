#!/usr/bin/env python
''' This script loads the tercile probabilities for a single model and forecast previously obtained with pred2tercile_operational.py, transforms them into categorical values (1,2,3) indicating the most probable tercile, adds its probability as well as a skill mask; these 3 data variables per variable are saved into a new netCDF file.'''

#load packages
from datetime import date
import numpy as np
import xarray as xr
import os
import pandas as pd
import dask
import sys
import pdb
import yaml
from pathlib import Path



# CREATE LIST CONTAINING ALL DOMAINS TO BE PROCESSED ######################################

domain_list = ['medcof','Iberia','Canarias']

####################################################################

for do in np.arange(len(domain_list)):

    ## CONSTRUCT CONFIGURATION FILE ######################################

    configuration_file = 'config_for_seas2ipe_'+domain_list[do]+'.yaml'

    ######################################################################

    def load_config(config_file='config/'+configuration_file):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / config_file
        print('The path of the configuration file is '+str(config_path))
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Setup paths based on GCM_STORE environment variable
        gcm_store = os.getenv('GCM_STORE', 'lustre')
        if gcm_store in config['paths']:
            paths = config['paths'][gcm_store]
            # Handle special cases for argo environment
            if gcm_store == 'argo':
                data_dir = os.getenv("DATA_DIR", "")
                paths['home'] = data_dir
                paths['path_gcm_base'] = data_dir + paths['path_gcm_base']
                paths['path_gcm_base_derived'] = data_dir + paths['path_gcm_base_derived']
                paths['path_gcm_base_masked'] = data_dir + paths['path_gcm_base_masked']
                paths['dir_forecast'] = data_dir + paths['dir_forecast']
            config['paths'] = paths
        else:
            raise Exception(f'ERROR: unknown entry for <gcm_store> !')

        return config

    # Load configuration
    config = load_config()

    #the init of the forecast (year and month) can passed by bash; if nothing is passed these parameters will be set by python
    if len(sys.argv) == 2:
        print("Reading from input parameters passed via bash")
        year_init = str(sys.argv[1])[0:4]
        month_init = str(sys.argv[1])[-2:]
        if len(year_init) != 4 or len(month_init) != 2:
            raise Exception('ERROR: check length of <year_month_init> input parameter !')
    else:
        print("No input parameter have been provided by the user and the script will set the <year_init> and <month_init> variables for the year and month of the current date...")
        year_init = str(date.today().year)
        month_init = f"{date.today().month:02d}"
        print(date.today())
    print(year_init, month_init)

    # # # Example year and run to run without passing any input arguments; comment or delete the next two lines in operative use
    # year_init = 2024 #a list containing the years the forecast are initialized on, will be looped through with yy
    # month_init = 10 #a list containing the corresponding months the forecast are initialized on, will be called while looping through <year_init> (with yy), i.e. must have the same length

    # Extract configuration variables
    vers = config['version']
    model = config['model']
    version = config['model_version']
    obs = config['obs']
    years_quantile = config['years_quantile']
    subperiod = config['subperiod']
    score = config['score']
    agg_labels = config['agg_labels']
    nan_placeholder = config['nan_placeholder']

    variables_std = config['variables_std']
    variables_out = config['variables_out']
    masked_variables_std = config['masked_variables_std']

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

    print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

    #create output directory of the forecasts generated here, if it does not exist.
    if os.path.isdir(dir_output) != True:
        os.makedirs(dir_output)

    for ag in np.arange(len(agg_labels)):
        for mm in np.arange(len(model)):
            #load the skill masks for a given aggregation window; multiple variables are and models located within the same file.
            filename_validation = dir_validation+'/skill_masks/'+domain+'/'+agg_labels[ag]+'/skill_masks_pticlima_'+domain+'_'+agg_labels[ag]+'_'+model[mm]+version[mm]+'_'+vers+'.nc'
            nc_val = xr.open_dataset(filename_validation)

            #check if model name in skill mask file matches the requested model name
            if nc_val.model != model[mm]+version[mm]:
                raise ValueError('The model name requested by <model[mm]> does not match the model name stored in <nc_val> !')

            nc_forecast = xr.Dataset() #create empty xarray dataset to be filled with xr data arrays in the next loop
            for vv in np.arange(len(variables_std)):
                #load the forecast for a specific variable
                filename_forecast = dir_forecast+'/'+domain+'/probability_'+agg_labels[ag]+'_'+model[mm]+version[mm]+'_'+variables_std[vv]+'_'+domain+'_init_'+str(year_init)+str(month_init).zfill(2)+'_dtr_'+detrended+'_refyears_'+str(years_quantile[mm][0])+'_'+str(years_quantile[mm][1])+'_'+vers+'.nc'
                nc_forecast_step = xr.open_dataset(filename_forecast) #get the xr data array containing the tercile probabilities for a specific variable

                #check whether the previously stored model and version thereof match those requested by this script
                if nc_forecast_step.model == model[mm]+version[mm]:
                    print('The requested model '+model[mm]+' and its version '+version[mm]+' coincide with the entry previously stored in '+filename_forecast+' !')
                else:
                    raise ValueError('The requested model '+model[mm]+' and its version '+version[mm]+' do NOT coincide with the entry previously stored in '+filename_forecast+' !')

                tercile_attrs = nc_forecast_step.tercile.attrs

                nc_forecast_step_prob = nc_forecast_step['probability'].sel(aggregation=agg_labels[ag],model=model[mm]+version[mm]).drop_vars(['aggregation','model'])

                #get maximum probability, #note that nc_forecast_step_prob only contains the probability of the most likely tercile ! The other two terciles are set to nan by pred2tercile_operational.py
                nc_forecast_step_maxprob = nc_forecast_step_prob.max(dim='tercile').astype(datatype)
                valid_data_bool_4d = ~np.isnan(nc_forecast_step_maxprob) #get valid data for the 4d array
                valid_data_bool_5d = ~np.isnan(nc_forecast_step_prob) #get valid data for the 5d array
                nc_forecast_step_argmax = nc_forecast_step_prob.where(valid_data_bool_5d, other=nan_placeholder).argmax(dim='tercile')+1 #set nan values to 0 and calculate tercile position of the maxiumum probability value (1,2 or 3)
                nc_forecast_step_argmax = nc_forecast_step_argmax.where(valid_data_bool_4d,other=np.nan) #set tercile position to nan where maxima are nan

                #clean
                nc_forecast_step_prob.close()
                del(nc_forecast_step_prob)

                nc_forecast['mlt_'+variables_out[vv]] = nc_forecast_step_argmax
                nc_forecast['prob_'+variables_out[vv]] = nc_forecast_step_maxprob
                #add attributes
                nc_forecast['mlt_'+variables_out[vv]].attrs['long_name'] = 'Tercile category (1=lower, 2=middle, 3=upper) for '+variables_out[vv]
                nc_forecast['mlt_'+variables_out[vv]].attrs['units'] = '1'
                nc_forecast['prob_'+variables_out[vv]].attrs['long_name'] = 'probability_of_most_likely_tercile_for_'+variables_out[vv]
                nc_forecast['prob_'+variables_out[vv]].attrs['standard_name'] = 'probability_of_event_in_category'
                nc_forecast['prob_'+variables_out[vv]].attrs['units'] = '1'
                #clean
                nc_forecast_step_argmax.close()
                nc_forecast_step_maxprob.close()
                nc_forecast_step.close()
                del(nc_forecast_step_argmax,nc_forecast_step_maxprob,nc_forecast_step)

                #filter subperiod, detrending option, model and version, season, and variable from skill mask
                fc_season_length = nc_forecast.season.length_in_months

                #check whether the requested aggregation window matches the season length stored within the file
                if fc_season_length != int(agg_labels[ag][0]):
                    raise ValueError('<fc_season_length> must equal <agg_labels[ag]> ! ')

                fc_seasons = nc_forecast.season.values
                nc_val_sub = nc_val.sel(subperiod=subperiod,detrended=detrended,season=fc_seasons,variable=variables_std[vv])[score]
                val_leads = nc_val_sub.lead.values

                skill_mask = np.zeros((len(fc_seasons),len(nc_val.y),len(nc_val.x))) #a 3d array
                for sea in np.arange(len(fc_seasons)):
                    skill_mask_step = nc_val_sub.sel(season=fc_seasons[sea],lead=val_leads[sea])
                    skill_mask[sea,:,:] = skill_mask_step.values

                #bring numpy array into xarray data array format
                skill_mask = xr.DataArray(skill_mask.astype('float32'),coords=[fc_seasons,nc_val.y,nc_val.x],dims=['season','y','x'], name=variables_std[vv]+'_skill_mask')

                #cut down the forecast domain to the skill domain indicated in the <domain> input variable
                nc_forecast = nc_forecast.sel(y=skill_mask.y,x=skill_mask.x)

                # apply land-sea mask to skill mask
                if variables_std[vv] in masked_variables_std:
                    print('Upon user request, values for sea grid-boxes are set to nan in skill mask '+variables_std[vv]+' ! ')
                    land_sea_3d = np.squeeze(land_sea_4d)
                    skill_mask = skill_mask.where(land_sea_3d == 1, other=np.nan)
                elif variables_std[vv] not in masked_variables_std:
                    print('As requested by the user, the verification results are not filtered by a land-sea mask in skill mask for '+variables_std[vv]+' !')
                else:
                    raise ValueError('check whether <variables_std[vv]> is in <masked_variables_std> !')

                # add attributes to skill mask
                skill_mask.attrs['long_name'] = 'binary_skill_mask_based_on_'+score+' for '+variables_out[vv]
                skill_mask.attrs['standard_name'] = 'binary_skill_mask'
                skill_mask.attrs['units'] = 'binary'

                # merge skill mask into the existing xarray dataset containing the most like tercile and its probability
                nc_forecast['skill_'+variables_out[vv]] = skill_mask #assign variable-specific skill mask to the newly generated xarray dataset produced by this script
                skill_mask.close()
                del(skill_mask)

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
            nc_val_sub.close()
            nc_forecast.close()
            del(nc_forecast,nc_val_sub)

        #clean up
        nc_val.close()
        del(nc_val)

print('INFO: seas2ipe.py has been run successfully ! The netcdf output file has been stored at '+savename)
quit()
