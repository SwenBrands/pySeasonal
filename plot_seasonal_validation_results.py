#!/usr/bin/env python

'''Plots verification results calculated before with get_skill_season.py .
Author: Swen Brands, brandssf@ifca.unican.es
'''

#load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import os
import pandas as pd
import xskillscore as xs
from math import radians, cos, sin, asin, sqrt #needed to calculate haversine distance

#set input parameters
model_dataset = ['ecmwf51'] #list of model or reanalysis datasets: era5 or era5_land
corr_outlier = 'no' #load the outlier-correted validation results; yes or no
detrending = ['yes','no'] #yes or no, linear detrending of the gcm and obs time series prior to validation
file_years = [1981,2022] #start and end years indicated in the input file name
file_system = 'lustre' #lustre or myLaptop; used to create the path structure to the input and output files

variables = ['SPEI-3','fwi','tp','ssrd','si10','t2m','msl'] #variable names in CDS format
ref_dataset = ['era5','era5','era5','era5','era5','era5','era5'] # #list of model or reference observational dataset paired with <variables> input parameter below

#variables = ['SPEI-3'] #variable names in CDS format
#ref_dataset = ['era5'] # #list of model or reference observational dataset paired with <variables> input parameter below

domain = 'medcof'
critval_rho = 0.05 #critical value used to define the signficance of the correlation measuere applied here (Pearon and Spearman)
critval_skillscore = 0 #threshold value above which the skill scores applied here indicate skill (is normally set to 0). Currently the climatological mean value of the reference dataset (e.g. ERA5) is used as naiv reference forecast: Skill Score = 1 - SCORE / SCORE_clim  
critval_relbias = 5 #percentage threshold beyond which the absolute relative bias is printed with a dot in the maps and thus assumed to be "important"
scores = ['relbias','spearman_r','pearson_r','crps_ensemble_skillscore_clim']
relbias_max = 100 #magnitude of the upper and lower limit to be plotted in case of relbias and tp, this is a percentage value and it is used because the relbias can be very large in dry regions due to the near-to-zero climatological precip. there
vers = 'as_input_file' # 'as_input_file' searches the version stored in the input files generated before with get_skill_season.py; other entries will be directly passed to the netCDF output file produced here

precision = 'float32' #precision of the variable in the output netCDF files
dpival = 300 #resultion of the output figure in dpi
figformat = 'pdf' #format of the output figures: pdf, png, etc.
colormap_ascend = 'Spectral_r' #ascendig colormap used for plotting: Spectral_r 
colormap_div = 'seismic' #diverging (zero-centered) colormap used for plotting: seismic
titlesize = 6

##EXECUTE ##############################################################

#set directory tree as a function of the file system
if file_system == 'myLaptop':
    home = os.getenv('HOME')
    rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal' #script directory, you should be there or point to this directory when running these scripts via python
    dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation' #path to outupt netcdf files produced with this script, containing an xarray dataset with all verification results
    dir_figs = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation' #path to output figures file generated by this script
    exec(open(rundir+'/functions_seasonal.py').read())
    exec(open(home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySolar/functions_radiation.py').read())
elif file_system == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    rundir = home+'/Inventory/Scripts/pyPTIclima/pySeasonal'
    dir_netcdf = home+'/Inventory/Results/seasonal/validation' #path to outupt netcdf files produced with this script, containing an xarray dataset with all verification results
    dir_figs = home+'/Inventory/Results/seasonal/validation' #path to output figures file generated by this script
    exec(open(rundir+'/functions_seasonal.py').read())
    exec(open(home+'/Inventory/Scripts/pyPTIclima/pySolar/functions_radiation.py').read())
else:
    raise Exception('ERROR: unknown entry for <file_system> input parameter!')

dir_maps = dir_figs+'/maps'

print('INFO: Verfying '+str(model_dataset)+' against for '+str(variables)+' from '+str(ref_dataset)+' domain '+domain+', detrending '+str(detrending)+' and outlier correction '+corr_outlier)
## check consistency of input parameters
if len(variables) != len(ref_dataset):
    raise Exception('ERROR: The two input lists <variables> and <ref_dataset> must have the same length !')

#create output directories if they do not exist.
if os.path.isdir(dir_figs) != True:
    os.makedirs(dir_figs)
for vv in np.arange(len(variables)):
    if os.path.isdir(dir_figs+'/'+variables[vv]) != True:
        os.makedirs(dir_figs+'/'+variables[vv])  

#init global minimum and maximum values
minvals_map = np.empty((len(detrending),len(variables),len(model_dataset),len(scores)))
minvals_map[:] = np.nan
maxvals_map = minvals_map.copy()
minvals_pcolor = minvals_map.copy()
maxvals_pcolor = minvals_map.copy()
for det in np.arange(len(detrending)):
    for vv in np.arange(len(variables)):
        if os.path.isdir(dir_figs+'/'+variables[vv]+'/maps') != True:
            os.makedirs(dir_figs+'/'+variables[vv]+'/maps')
        for dd in range(len(model_dataset)):
            colormaps = [] #the colormap is the same for each dataset and only depends on the score so this is a list with a length equalling the length of <scores>
            #load netcdf files containing the verification results
            if model_dataset[dd] == 'ecmwf51':
                filename_input = 'validation_results_season_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending[det]+'_'+str(file_years[0])+'_'+str(file_years[1])+'.nc'
            else:
                raise Exception('ERROR: unknown entry for <model_dataset> !')
            nc_results = xr.open_dataset(dir_netcdf+'/'+filename_input)
            
            #extract the min and max values for each score, <map> prefix points to values used for mapping at the grid-box scale and <pcolor> points to the values used in the pcolor figure
            for sc in np.arange(len(scores)):
                if scores[sc] in ('bias','relbias'):
                    if scores[sc] == 'relbias' and variables[vv] in ('tp','si10'):
                        #print('INFO: for '+variables[vv]+' and '+scores[sc]+', the '+str(prct_cb_precip)+' percentile defines the limit of the colormap in the error maps to be plotted because the maximum absolute '+scores[sc]+' is very large in very dry regions due to near-to-zero climatological precipitation there.')
                        #maxvals_map[det,vv,dd,sc] = np.percentile(np.abs(nc_results[scores[sc]]),prct_cb_precip) #a high percentile is here used as upper limit of the color bar because the maximum relative bias can be very large. In very dry regions with near-to-zero precip. climatologies, a small absolute deviation can lead to a very large relative bias
                        #minvals_map[det,vv,dd,sc] = np.percentile(np.abs(nc_results[scores[sc]]),prct_cb_precip)*-1
                        print('INFO: for '+variables[vv]+' and '+scores[sc]+', no min and max values are stored because they can be very large in dry regions due to the near-to-zero climatolologial precipitation.')
                    else:
                        maxvals_map[det,vv,dd,sc] = np.abs(nc_results[scores[sc]]).max().values
                        minvals_map[det,vv,dd,sc] = np.abs(nc_results[scores[sc]]).max().values*-1
                        maxvals_pcolor[det,vv,dd,sc] = np.abs(nc_results[scores[sc]].mean(dim='x').mean(dim='y')).max().values #maximum of the seasonal areal mean values
                        minvals_pcolor[det,vv,dd,sc] = np.abs(nc_results[scores[sc]].mean(dim='x').mean(dim='y')).max().values*-1 #minimum of the seasonal areal mean values
                    colormaps.append(colormap_div)
                elif scores[sc] in ('mae','mape','rmse','crps_ensemble'):
                    maxvals_map[det,vv,dd,sc] = nc_results[scores[sc]].max().values #global maximum for the specific model dataset and variable
                    minvals_map[det,vv,dd,sc] = nc_results[scores[sc]].min().values #global minimum for the specific model dataset and variable
                    maxvals_pcolor[det,vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').max().values #maximum of the seasonal areal mean values
                    minvals_pcolor[det,vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').min().values #minimum of the seasonal areal mean values
                    colormaps.append(colormap_ascend)
                elif scores[sc] in ('crps_ensemble_skillscore_clim'):
                    maxvals_map[det,vv,dd,sc] = nc_results[scores[sc]].max().values
                    minvals_map[det,vv,dd,sc] = 0
                    percent_sig, sig_arr = get_frac_above_threshold(nc_results[scores[sc]].values.copy(),0) #percent_sig here refers to the percentage of grid-boxes with skill score values > 0
                    maxvals_pcolor[det,vv,dd,sc] = np.max(percent_sig)
                    minvals_pcolor[det,vv,dd,sc] = 0
                    colormaps.append(colormap_ascend)
                elif scores[sc] in ('pearson_r','spearman_r'):
                    if scores[sc] in ('pearson_r'):
                        score_pval = 'pearson_pval'
                    elif scores[sc] in ('spearman_r'):
                        score_pval = 'spearman_pval'
                    else:
                        raise Exception('ERROR: check entry in <scores[sc]> !')
                    maxvals_map[det,vv,dd,sc] = nc_results[scores[sc]].max().values
                    minvals_map[det,vv,dd,sc] = 0
                    percent_sig, sig_arr = get_frac_significance(nc_results[score_pval].values,nc_results[scores[sc]].values,critval_rho)
                    maxvals_pcolor[det,vv,dd,sc] = np.max(percent_sig)
                    minvals_pcolor[det,vv,dd,sc] = 0
                    colormaps.append(colormap_ascend)
                else:
                    raise Exception('ERROR: unknown value for <scores[sc]> !')
            ##close file
            nc_results.close()
            del(nc_results,score_pval)
    
### get global min and max values covering the range of all considered model / reanalysis datasets, currently does not take into account various GCM datasets, i.e. only works for len(model_dataset) == 1
# minvals_pcolor = np.reshape(minvals_pcolor,(minvals_pcolor.shape[0],minvals_pcolor.shape[1]*minvals_pcolor.shape[2],minvals_pcolor.shape[3])).min(axis=1)
# maxvals_pcolor = np.reshape(maxvals_pcolor,(maxvals_pcolor.shape[0],maxvals_pcolor.shape[1]*maxvals_pcolor.shape[2],maxvals_pcolor.shape[3])).max(axis=1)
# minvals_map = np.reshape(minvals_map,(minvals_map.shape[0],minvals_map.shape[1]*minvals_map.shape[2],minvals_map.shape[3])).min(axis=1)
# maxvals_map = np.reshape(maxvals_map,(maxvals_map.shape[0],maxvals_map.shape[1]*maxvals_map.shape[2],maxvals_map.shape[3])).max(axis=1)

# minvals_pcolor = np.reshape(minvals_pcolor,(minvals_pcolor.shape[0]*minvals_pcolor.shape[1]*minvals_pcolor.shape[2],minvals_pcolor.shape[3])).min(axis=0)
# maxvals_pcolor = np.reshape(maxvals_pcolor,(maxvals_pcolor.shape[0]*maxvals_pcolor.shape[1]*maxvals_pcolor.shape[2],maxvals_pcolor.shape[3])).max(axis=0)
# minvals_map = np.reshape(minvals_map,(minvals_map.shape[0]*minvals_map.shape[1]*minvals_map.shape[2],minvals_map.shape[3])).min(axis=0)
# maxvals_map = np.reshape(maxvals_map,(maxvals_map.shape[0]*maxvals_map.shape[1]*maxvals_map.shape[2],maxvals_map.shape[3])).max(axis=0)

minvals_pcolor = np.nanmin(np.reshape(minvals_pcolor,(minvals_pcolor.shape[0]*minvals_pcolor.shape[1]*minvals_pcolor.shape[2],minvals_pcolor.shape[3])),axis=0)
maxvals_pcolor = np.nanmax(np.reshape(maxvals_pcolor,(maxvals_pcolor.shape[0]*maxvals_pcolor.shape[1]*maxvals_pcolor.shape[2],maxvals_pcolor.shape[3])),axis=0)
minvals_map = np.nanmin(np.reshape(minvals_map,(minvals_map.shape[0]*minvals_map.shape[1]*minvals_map.shape[2],minvals_map.shape[3])),axis=0)
maxvals_map = np.nanmax(np.reshape(maxvals_map,(maxvals_map.shape[0]*maxvals_map.shape[1]*maxvals_map.shape[2],maxvals_map.shape[3])),axis=0)

#then plot the results with this min and max values
for det in np.arange(len(detrending)):
    for vv in np.arange(len(variables)):
        for dd in range(len(model_dataset)):
            #load netcdf files containing the verification results
            if model_dataset[dd] == 'ecmwf51':
                #verification_results_season_tp_ecmwf51_vs_era5_medcof_corroutlier_no_detrended_no_1981_2022.nc
                filename_input = 'validation_results_season_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending[det]+'_'+str(file_years[0])+'_'+str(file_years[1])+'.nc'
            else:
                raise Exception('ERROR: unknown entry for <model_dataset> !')
            nc_results = xr.open_dataset(dir_netcdf+'/'+filename_input)
            #initializing output numpy matrix containing a binary resutls array (1 = significant skill, 0 = spurious skill)
            if det == 0 and vv == 0 and dd == 0:
                #get meshes for plotting the maps and init the output binary mask; halfres is used for plotting maps below
                y_coord = nc_results.y.values
                x_coord = nc_results.x.values
                xx,yy = np.meshgrid(x_coord,y_coord)
                binary_mask = np.zeros((len(detrending),len(variables),len(model_dataset),len(scores),len(nc_results.season),len(nc_results.lead),len(y_coord),len(x_coord)),dtype='single')
                binary_mask[:] = np.nan
                halfres = np.abs(np.diff(nc_results.x.values))[0]/2 #needed to plot the pcolormesh

            ##plot matrices of verification results for the distinct score (x-axis = seasons, y-axis = stations and save to <figformat>
            score_unit = np.zeros((len(scores))).tolist()
            score_info = score_unit.copy()
            for sc in np.arange(len(scores)):
                print('INFO: plotting '+scores[sc]+'...')
                score_ref = nc_results.reference_observations
                if os.path.isdir(dir_figs+'/'+variables[vv]+'/maps/'+scores[sc]) != True:
                    os.makedirs(dir_figs+'/'+variables[vv]+'/maps/'+scores[sc])
                if scores[sc] in ('crps_ensemble_skillscore_clim'):
                    #areal percentage of positive skill score values
                    savename = dir_figs+'/'+variables[vv]+'/pcolor_posgridboxes_'+variables[vv]+'_'+scores[sc]+'_'+domain+'_'+model_dataset[dd]+'_vs_'+score_ref+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending[det]+'_testlvl_'+str(round(critval_rho*100))+'.'+figformat
                    #pval = nc_results['pearson_pval'].values
                    #rho = nc_results['pearson_r'].values
                    pcolorme, sig_arr = get_frac_above_threshold(nc_results[scores[sc]].values.copy(),0) #if .copy() is not put here, then nc_results[scores[sc]] is overwritten, which is a very strange bug !
                    units_pcolor = '-inf <= score <= 1'
                    label_pcolor = 'areal fraction of positive '+scores[sc]
                    mapme = nc_results[scores[sc]].values
                    binmask = np.zeros(mapme.shape)
                    binmask[:] = np.nan
                    #mask1 = mapme > critval_skillscore
                    #mask0 = mapme <= critval_skillscore
                    mask1 = mapme > 0
                    mask0 = mapme <= 0
                    binmask[mask1] = 1
                    binmask[mask0] = 0
                    score_unit[sc] = 'binary'
                    score_info[sc] = 'Continuous Rank Probabililty Score with reference to climatological forecast exceeding '+str(critval_skillscore)+', yes (1) or no (0); the shape of the continuous variable distribution is taken from the ensemble members'
                elif scores[sc] in ('pearson_r','spearman_r'):
                    if scores[sc] in ('pearson_r'):
                        score_pval = 'pearson_pval'
                    elif scores[sc] in ('spearman_r'):
                        score_pval = 'spearman_pval'
                    else:
                        raise Exception('ERROR: check entry in <scores[sc]> !')
                    #areal percentage of significant grid-box scale correlation coefficients is calculated and plotted
                    savename = dir_figs+'/'+variables[vv]+'/pcolor_siggridboxes_'+variables[vv]+'_'+scores[sc]+'_'+domain+'_'+model_dataset[dd]+'_vs_'+score_ref+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending[det]+'_testlvl_'+str(round(critval_rho*100))+'.'+figformat
                    pval = nc_results[score_pval].values
                    rho = nc_results[scores[sc]].values
                    pcolorme, sig_arr = get_frac_significance(pval.copy(),rho.copy(),critval_rho) #.copy() is mandatory here; otherwise pval will be changed within the get_frac_significance() function
                    units_pcolor = '%'
                    label_pcolor = 'areal fraction of sig. positive '+scores[sc]
                    mapme = rho
                    binmask = np.zeros(mapme.shape)
                    binmask[:] = np.nan
                    mask1 = (pval < critval_rho) & (rho > 0)
                    mask0 = (pval >= critval_rho) | (rho <= 0)
                    binmask[mask1] = 1
                    binmask[mask0] = 0
                    score_unit[sc] = 'binary'
                    score_info[sc] = 'significant '+scores[sc]+' at '+str(round(critval_rho*100))+' percent test-level and positive sign for the ensemble mean time series, yes (1) or no (0)'
                #currently the 10th percentile of the cprs is assumed to set the binary mask to 1 (if below this percentile)
                elif scores[sc] in ('crps_ensemble'):
                    savename = dir_figs+'/'+variables[vv]+'/pcolor_arealmean_'+variables[vv]+'_'+scores[sc]+'_'+domain+'_'+model_dataset[dd]+'_vs_'+score_ref+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending[det]+'.'+figformat
                    pcolorme = nc_results[scores[sc]].mean(dim='y').mean(dim='x')
                    units_pcolor = nc_results[scores[sc]].units
                    label_pcolor = 'areal mean '+nc_results[scores[sc]].name
                    mapme = nc_results[scores[sc]].values
                    binmask = np.zeros(mapme.shape)
                    binmask[:] = np.nan
                    threshold = np.percentile(mapme.flatten(),10)
                    mask1 = mapme < threshold
                    mask0 = mapme >= threshold
                    binmask[mask1] = 1
                    binmask[mask0] = 0
                    #score_unit[sc] =  scores[sc]+' can range in between 0 <= '+scores[sc]+' < +infinity; here: '+scores[sc]+' < 10th percentile yes (1) or no (0)'
                    score_unit[sc] = 'binary'
                    score_info[sc] = 'Continuous Rank Probabililty Score exceeding '+critval_skillscore+', yes (1) or no (0); the shape of the continuous variable distribution is taken from the ensemble members'
                elif scores[sc] in ('relbias'):
                    savename = dir_figs+'/'+variables[vv]+'/pcolor_arealmean_'+variables[vv]+'_'+scores[sc]+'_'+domain+'_'+model_dataset[dd]+'_vs_'+score_ref+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending[det]+'.'+figformat
                    pcolorme = nc_results[scores[sc]].mean(dim='y').mean(dim='x')
                    units_pcolor = nc_results[scores[sc]].units
                    label_pcolor = 'areal mean '+nc_results[scores[sc]].name
                    mapme = nc_results[scores[sc]].values
                    binmask = np.zeros(mapme.shape)
                    binmask[:] = np.nan
                    threshold = critval_relbias #this is a percentage threshold
                    mask1 = np.abs(mapme) < threshold
                    mask0 = np.abs(mapme) >= threshold
                    binmask[mask1] = 1
                    binmask[mask0] = 0
                    score_unit[sc] = 'binary'
                    score_info[sc] = 'relative bias of the ensemble mean time series in percent of the observed mean value exceeding '+str(critval_relbias)+', yes (1) or no (0)'
                else:
                    raise Exception('ERROR: '+scores[sc]+' are currently not supported by plot_seasonal_validation_results.py !')
                
                #convert to xr dataArray and add metadata necessary for plotting
                pcolorme = xr.DataArray(pcolorme,coords=[np.arange(len(nc_results.season.values)),np.arange(len(nc_results.lead.values))],dims=['season', 'lead'], name=label_pcolor)
                pcolorme.attrs['units'] = units_pcolor
                pcolorme.attrs['season_label'] = nc_results.season.values
                pcolorme.attrs['lead_label'] = nc_results.lead.values
                if scores[sc] == 'relbias' and variables[vv] in ('tp','si10'):
                    plot_pcolormesh_seasonal(pcolorme,relbias_max*-1,relbias_max,savename,colormaps[sc],dpival) #colormap limits were set by the user in <relbias_max>
                else:
                    #plot_pcolormesh_seasonal(pcolorme,minvals_pcolor[det,sc],maxvals_pcolor[det,sc],savename,colormaps[sc],dpival) #colormap limits have been inferred from the score arrays above
                    plot_pcolormesh_seasonal(pcolorme,minvals_pcolor[sc],maxvals_pcolor[sc],savename,colormaps[sc],dpival)
                
                #produce a map for each variable, model dataset season and lead
                seasons = nc_results.season.values
                leads = nc_results.lead.values
                if scores[sc] in ('relbias','bias') and detrending[det] == 'yes':
                    print('INFO: '+scores[sc]+' is not plotted for detrended time series since the intercept/mean is also removed by the detrending function and the bias is thus 0 by definition.')
                else:
                    for sea in np.arange(len(seasons)):
                        for ll in np.arange(len(leads)):
                            if scores[sc] in ('pearson_r','spearman_r'):
                                critval_label = str(round(critval_rho*100))
                            elif scores[sc] in ('relbias'):
                                critval_label = str(critval_relbias)
                            elif scores[sc] in ('crps_ensemble_skillscore_clim'):
                                critval_label = str(critval_skillscore)
                            else:
                                raise Exception('ERROR: '+scores[sc]+' is not yet supported by this function !')
                            title = variables[vv]+' '+seasons[sea]+' '+leads[ll]+' '+scores[sc]+' '+critval_label+' dtr'+detrending[det]+' '+domain+' '+model_dataset[dd]+' vs '+score_ref+' '+str(file_years[0])+' '+str(file_years[1])
                            savename = dir_figs+'/'+variables[vv]+'/maps/'+scores[sc]+'/map_'+variables[vv]+'_'+seasons[sea]+'_'+leads[ll]+'_'+scores[sc]+'_'+critval_label+'_'+domain+'_'+model_dataset[dd]+'_vs_'+score_ref+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending[det]+'_'+str(file_years[0])+str(file_years[1])+'.'+figformat
                            cbarlabel = scores[sc]
                            agreeind = binmask[sea,ll,:,:] == 1
                            if scores[sc] == 'relbias' and variables[vv] in ('tp','si10','msl'):
                                get_map_lowfreq_var(mapme[sea,ll,:,:],xx,yy,agreeind,relbias_max*-1,relbias_max,dpival,title,savename,halfres,colormaps[sc],titlesize,cbarlabel) #colormap limits were set by the user in <relbias_max>
                            else:
                                #get_map_lowfreq_var(mapme[sea,ll,:,:],xx,yy,agreeind,minvals_map[det,sc],maxvals_map[det,sc],dpival,title,savename,halfres,colormaps[sc],titlesize,cbarlabel) #colormap limits have been inferred from the score arrays above
                                get_map_lowfreq_var(mapme[sea,ll,:,:],xx,yy,agreeind,minvals_map[sc],maxvals_map[sc],dpival,title,savename,halfres,colormaps[sc],titlesize,cbarlabel)
                
                #save a binary mask (significance or skill yes or no) in netcdf format
                binary_mask[det,vv,dd,sc,:,:,:,:] = binmask
            ##close input nc files and produced xr dataset
            pcolorme.close()
        
        #bring the binary result masks into xarray data array format, one array per score. Then assign metadata to score / dataarray and stack them all as variables into a definite xarray dataset to be stored on netCDF
        for sc in np.arange(len(scores)):
            if scores[sc] in ('bias','relbias','pearson_r','spearman_r','crps_ensemble_skillscore_clim'):
                print('INFO: saving validation results for '+scores[sc])
                binary_mask_score_i = binary_mask[:,:,:,sc,:,:,:,:]
                binary_mask_score_i = xr.DataArray(binary_mask_score_i,coords=[detrending,variables,model_dataset,nc_results.season,nc_results.lead,nc_results.y,nc_results.x],dims=['detrended','variable', 'model', 'season', 'lead', 'y', 'x'], name=scores[sc]+'_binary')
                binary_mask_score_i['detrended'].attrs['info'] = 'Linear de-trending was applied to the modelled and (quasi)observed time series prior to validation; yes or no'
                binary_mask_score_i['variable'].attrs['info'] = 'Meteorological variable acronym according to ERA5 nomenclature followed by Copernicus Climate Data Store (CDS)'
                binary_mask_score_i['model'].attrs['info'] = 'Name and version of the model / prediction system'
                binary_mask_score_i['season'].attrs['info'] = 'Season the forecast is valid for'
                binary_mask_score_i['lead'].attrs['info'] = 'Leadtime of the forecast; one per month'
                binary_mask_score_i['y'].attrs['name'] = 'latitude'
                binary_mask_score_i['y'].attrs['standard_name'] = 'latitude'
                binary_mask_score_i['x'].attrs['name'] = 'longitude'
                binary_mask_score_i['x'].attrs['standard_name'] = 'longitude'
                binary_mask_score_i.attrs['info'] = score_info[sc]
                binary_mask_score_i.attrs['unit'] = score_unit[sc]
                binary_mask_score_i.attrs['unit'] = score_unit[sc]
                if sc == 0:
                    ds_binary_mask = binary_mask_score_i.to_dataset()
                else:
                    ds_binary_mask[scores[sc]+'_binary'] = binary_mask_score_i
                binary_mask_score_i.close()
                del(binary_mask_score_i)
                ds_binary_mask.attrs['author'] = "Swen Brands (CSIC-UC, Instituto de Fisica de Cantabria), brandssf@ifca.unican.es or swen.brands@gmail.com"
                ds_binary_mask.attrs['validation_period'] = str(file_years[0])+' to '+str(file_years[1])
                #set version of the netCDF output file
                if vers == 'as_input_file':
                    version_label = nc_results.version
                else:
                    version_label = vers
                ds_binary_mask.attrs['version'] = version_label
                ds_binary_mask.attrs['nan_criterion'] = 'A nan is set at a given grid-box if it is returned by xskillscore, e.g. due to a division by zero. It has been confirmed that this occcurs, e.g., if it does not rain at all either in the modelled or quasi-observed time-series.'
            else:
                print('WARNING: Validation results for '+scores[sc]+' are not yet saved to netCDF because the transition to binary format still has to be discussed with the other PTI members.')
                continue
        nc_results.close()
        savename_netcdf = dir_netcdf+'/binary_validation_results_pticlima_'+domain+'_'+str(file_years[0])+'_'+str(file_years[1])+'_v'+version_label+'.nc'
        ds_binary_mask.to_netcdf(savename_netcdf)
        ds_binary_mask.close()
        print('INFO: plot_seasonal_validation_results.py has been run successfully and results have been stores in netCDF format at:')
        print(savename_netcdf)
