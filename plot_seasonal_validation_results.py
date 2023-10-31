#!/usr/bin/env python

'''Verifies reanalysis data against AEMET station data, both brought into a comparable and harmonized format using 1. csv2nc.py and 2. get_neighbour.py prior to calling 3. validate.py (i.e. this script).
Author: Swen Brands, brandssf@ifca.unican.es
'''

#load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import os
import pandas as pd
import xskillscore as xs
from math import radians, cos, sin, asin, sqrt #needed to calculate haversine distance
home = os.getenv('HOME')
exec(open('functions_seasonal.py').read())
exec(open(home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySolar/functions_radiation.py').read())

#set input parameters
model_dataset = ['ecmwf51'] #list of model or reanalysis datasets: era5 or era5_land
ref_dataset = ['era5'] # #list of model or reference observational dataset paired with to <model_dataset>: aemet
rundir = home+'/datos/tareas/proyectos/pticlima/pyPTIclima/pySeasonal' #script directory, you should be there or point to this directory when running these scripts via python
dir_netcdf = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation' #path to outupt netcdf files produced with this script, containing an xarray dataset with all verification results
dir_figs = home+'/datos/tareas/proyectos/pticlima/seasonal/results/validation' #path to output figures file generated by this script
corr_outlier = 'no' #load the outlier-correted validation results; yes or no
detrending = 'yes' #yes or no, linear detrending of the gcm and obs time series prior to validation
file_years = [1981,2022] #start and end years indicated in the input file name
variables = ['t2m']
domain = 'medcof'
critval = 0.05

#scores = ['pearson_r','pearson_pval','pearson_pval_effn','spearman_r','spearman_pval','spearman_pval_effn'] #the scores to be plotted
scores = ['pearson_pval','spearman_pval','crps_ensemble']

precision = 'float32' #precision of the variable in the output netCDF files
dpival = 300 #resultion of the output figure in dpi
figformat = 'pdf' #format of the output figures: pdf, png, etc.
colormap_ascend = 'Spectral_r' #ascendig colormap used for plotting: Spectral_r 
colormap_div = 'seismic' #diverging (zero-centered) colormap used for plotting: seismic

##EXECUTE ##############################################################
# #check consistency of input parameters
# if len(variables) < 2:
    # raise Exception('ERROR: At least two variables must be evaluated for this script to work! Check entries for <variables>.')

#create output directories ir they do not exist.
if os.path.isdir(dir_figs) != True:
    os.makedirs(dir_figs)

#init global minimum and maximum values
minvals_map = np.empty((len(variables),len(model_dataset),len(scores)))
maxvals_map = np.empty((len(variables),len(model_dataset),len(scores)))
minvals_pcolor = np.empty((len(variables),len(model_dataset),len(scores)))
maxvals_pcolor = np.empty((len(variables),len(model_dataset),len(scores)))
for vv in np.arange(len(variables)):
    for dd in range(len(model_dataset)):
        colormaps = [] #the colormap is the same for each dataset and only depends on the score so this is a list with a length equalling the length of <scores>
        #load netcdf files containing the verification results
        if model_dataset[dd] == 'ecmwf51':
            #verification_results_season_tp_ecmwf51_vs_era5_medcof_corroutlier_no_detrended_no_1981_2022.nc
            filename_results = 'verification_results_season_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending+'_'+str(file_years[0])+'_'+str(file_years[1])+'.nc'
        else:
            raise Exception('ERROR: unknown entry for <model_dataset> !')
        nc_results = xr.open_dataset(dir_netcdf+'/'+filename_results)
        
        #extract the min and max values for each score, <map> prefix points to values used for mapping at the grid-box scale and <pcolor> points to the values used in the pcolor figure
        for sc in np.arange(len(scores)):
            if scores[sc] in ('bias','relbias'):
                maxvals_map[vv,dd,sc] = np.abs(nc_results[scores[sc]]).max().values
                minvals_map[vv,dd,sc] = np.abs(nc_results[scores[sc]]).max().values*-1
                maxvals_pcolor[vv,dd,sc] = np.abs(nc_results[scores[sc]].mean(dim='x').mean(dim='y')).max().values #maximum of the seasonal areal mean values
                minvals_pcolor[vv,dd,sc] = np.abs(nc_results[scores[sc]].mean(dim='x').mean(dim='y')).max().values*-1 #minimum of the seasonal areal mean values
                colormaps.append(colormap_div)
            elif scores[sc] in ('mae','mape','rmse','crps_ensemble'):
                maxvals_map[vv,dd,sc] = nc_results[scores[sc]].max().values #global maximum for the specific model dataset and variable
                minvals_map[vv,dd,sc] = nc_results[scores[sc]].min().values #global minimum for the specific model dataset and variable
                maxvals_pcolor[vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').max().values #maximum of the seasonal areal mean values
                minvals_pcolor[vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').min().values #minimum of the seasonal areal mean values
                colormaps.append(colormap_ascend)
            elif scores[sc] in ('pearson_pval','pearson_pval_effn','spearman_pval','spearman_pval_effn'):
                percent_sig = get_fraq_significance(nc_results[scores[sc]].values,critval)
                maxvals_pcolor[vv,dd,sc] = np.max(percent_sig)
                minvals_pcolor[vv,dd,sc] = 0
                colormaps.append(colormap_ascend)
            elif scores[sc] in ('pearson_r','spearman_r'):
                maxvals_map[vv,dd,sc] = nc_results[scores[sc]].max().values
                minvals_map[vv,dd,sc] = nc_results[scores[sc]].min().values
                maxvals_pcolor[vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').max().values #maximum of the seasonal areal mean values
                minvals_pcolor[vv,dd,sc] = nc_results[scores[sc]].mean(dim='x').mean(dim='y').min().values #minimum of the seasonal areal mean values
                colormaps.append(colormap_ascend)
            else:
                raise Exception('ERROR: unknown value for <scores[sc]> !')

        ##close file
        nc_results.close()
    
##get global min and max values covering the range of all considered model / reanalysis datasets, currently does not take into account various GCM datasets, i.e. only works for len(model_dataset) == 1
minvals_pcolor = np.reshape(minvals_pcolor,(minvals_pcolor.shape[0]*minvals_pcolor.shape[1],minvals_pcolor.shape[2])).min(axis=0)
maxvals_pcolor = np.reshape(maxvals_pcolor,(maxvals_pcolor.shape[0]*maxvals_pcolor.shape[1],maxvals_pcolor.shape[2])).max(axis=0)

#then plot the results with this min and max values
for vv in np.arange(len(variables)):
    for dd in range(len(model_dataset)):
        #load netcdf files containing the verification results
        if model_dataset[dd] == 'ecmwf51':
            #verification_results_season_tp_ecmwf51_vs_era5_medcof_corroutlier_no_detrended_no_1981_2022.nc
            filename_results = 'verification_results_season_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_'+domain+'_corroutlier_'+corr_outlier+'_detrended_'+detrending+'_'+str(file_years[0])+'_'+str(file_years[1])+'.nc'
        else:
            raise Exception('ERROR: unknown entry for <model_dataset> !')
        nc_results = xr.open_dataset(dir_netcdf+'/'+filename_results)        

        ##plot matrices of verification results for the distinct score (x-axis = seasons, y-axis = stations and save to <figformat>
        for sc in np.arange(len(scores)):
            print('INFO: plotting '+scores[sc]+'...')
            if scores[sc] in ('pearson_pval','pearson_pval_effn','spearman_pval','spearman_pval_effn'):
                #areal percentage of significant grid-box scale correlation coefficients is calculated and plotted
                savename = dir_figs+'/sig_gridboxes_'+scores[sc]+'_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending+'_testlvl_'+str(round(critval*100))+'.'+figformat
                plotme = get_fraq_significance(nc_results[scores[sc]].values,critval)
            elif scores[sc] in ('crps_ensemble'):
                #areal mean score is calculated and plotted
                savename = dir_figs+'/sig_gridboxes_'+scores[sc]+'_'+variables[vv]+'_'+model_dataset[dd]+'_vs_'+ref_dataset[dd]+'_corr_outlier_'+corr_outlier+'_detrended_'+detrending+'.'+figformat
                plotme = nc_results[scores[sc]].mean(dim='y').mean(dim='x')
            else:
                raise Exception('ERROR: '+scores[sc]+' are currently not supported by plot_seasonal_validation_results.py !')
            #convert to xr dataArray and add metadata necessary for plotting
            plotme = xr.DataArray(plotme,coords=[np.arange(len(nc_results.season.values)),np.arange(len(nc_results.lead.values))],dims=['season', 'lead'], name=scores[sc])
            plotme.attrs['units'] = nc_results[scores[sc]].units
            plotme.attrs['season_label'] = nc_results.season.values
            plotme.attrs['lead_label'] = nc_results.lead.values
            plot_pcolormesh_seasonal(plotme,minvals_pcolor[sc],maxvals_pcolor[sc],savename,colormaps[sc],dpival)

    ##close input nc files and produced xr dataset
    nc_results.close()
    plotme.close()
    print('INFO: plot_seasonal_validation_results.py has been run successfully !')
