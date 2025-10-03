#!/usr/bin/env python

## setup for various models
model = ['eccc','cmcc','ecmwf'] #seasonal forecast model
version = ['5','35','51'] #and version thereof; pertains to <model> loop indicated with <mm> below
n_mem = [20,40,25] #number of considered ensemble members, pertains to <model> loop. For instance, ECMWF51 has 25 hindcast and 51 forecast members so <n_mem = 25> should be set if hindcasts and forecasts are combined in the skill evaluation. The first n_mem members are selected.
n_lead = [8,7,8] #considered lead-time in months, pertains to <model> loop indicated with <mm> below, CMCC 3.5 hindcasts provide 183 forecast days, SEAS5.1 provide 215, ECCC provide 214

variables = [['pvpot','fwi','pr','tas','psl','sfcWind','rsds'],['pvpot','fwi','SPEI-3-M','pr','tas','psl','sfcWind','rsds'],['pvpot','fwi','SPEI-3-M','pr','tas','psl','sfcWind','rsds']] #variable names in directories and file names
variables_nc = [['pvpot','FWI','pr','tas','psl','sfcWind','rsds'],['pvpot','FWI','SPEI-3-M','pr','tas','psl','sfcWind','rsds'],['pvpot','FWI','SPEI-3-M','pr','tas','psl','sfcWind','rsds']] #variable names within the netCDF files, differs in case of fwi msi (msi / FWI is used within the file, but psl / fwi is used in the file name)
variables_new = [['pvpot','fwi','tp','t2m','msl','si10','ssrd'],['pvpot','fwi','SPEI-3-M','tp','t2m','msl','si10','ssrd'],['pvpot','fwi','SPEI-3-M','tp','t2m','msl','si10','ssrd']] #new variable names; as provided by ERA5 data from CDS
time_name = [['time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'],['time','time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'],['time','time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time']] #list containing lists, each sublist specifies the names of the time coordinates for each variables of a specific model; one list per model
lon_name = [['lon','lon','x','x','x','x','x'],['lon','lon','lon','x','x','x','x','x'],['lon','lon','lon','x','x','x','x','x']] #as time_name but for the name of the longitude coordinate
lat_name = [['lat','lat','y','y','y','y','y'],['lat','lat','lat','y','y','y','y','y'],['lat','lat','lat','y','y','y','y','y']] #as time_name but for the name of the latitude coordinate
# file_start = [['seasonal-original-single-levels_derived','seasonal-original-single-levels_derived','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels_derived','seasonal-original-single-levels_derived','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels_derived','seasonal-original-single-levels_derived','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels']] ##as time_name but for the name of the head folder containing hindcasts and predictions
file_start = [['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels']] ##as time_name but for the name of the head folder containing hindcasts and predictions


years = [[1993,2023],[1993,2023],[1981,2023]] #years to be regridded, the output files will be filled with monthly values, aggregated from daily values in this script, covering all months beginning in January of the indicated start year and ending in December of the indicated end year. If no daily input data is found for a given month, nans will be placed in the monthly output netCDF files.

#fixed parameters not passing through loops
imonth = [1,2,3,4,5,6,7,8,9,10,11,12] #month the forecasts are initialized on, 1 refers to the January 1st, 2 to Febrary 1st etc.

#set MEDCOF domain
domain = 'medcof' #spatial domain the model data is available on. So far, this is just a label used to find the input files and name the output files.
save_corrected_files = 'no' #overwrite input nc files with un expected units with newly generated corrected files

#setup for path to the input GCM files
gcm_store = 'lustre' #laptop, F, lustre or extdisk2

