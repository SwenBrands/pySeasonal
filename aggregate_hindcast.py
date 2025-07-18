#!/usr/bin/env python

'''This script aggregates and harmonizes daily GCM data arrays with the 4 dimensions time, member, lat, lon (or similar) to monthly-mean data arrays with the 5 dimensions time, lead, member, lat, lon and saves a newly generated xarray DataArray to netCDF format.
This is done to compare these monthly GCM values with monthly observations, regridded to the GCM grid with the <regrid_obs.py> script located in the same directory. The length of the time dimension in the output file
is 12 (months) x number of years set in the <years> parameter plus the number of forecast months (e.g. +8, 215 days, in case of ECMWF SEA5.1 from CDS). If an init month is not set in the <imonth> parameter (e.g. because it is not available), then the corresponding variable value is set to nan in the output file.
Forecasts are used if no hindcasts are available for a given year (e.g. for ECMWF51 hindcasts are availble unitl 2016 only). The range of considered years is set in the <years> parameter.
Note that the monthly mean-value of the last lead-time (set in <n_lead>) may be calculated on very few daily values and in this case should be discared aferwareds in the <get_skill.py> script contained in the same folder.
Author: Swen Brands, brandssf@ifca.unican.es'''

#load packages
import numpy as np
import xarray as xr
import dask
import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

# #setup for a single model
# model = ['ecmwf'] #seasonal forecast model
# version = ['51'] #and version thereof; pertains to <model> loop indicated with <mm> below
# time_name = [['time']] #one entry per model and variable; recurrs the <model> and <variables> loops below; one list per model, the list items refer to the time name set for the variable from that model; name of the time dimension in the netCDF files for this variable, corresponds to <variables> input parameter and must have the same length
# n_mem = [25] #number of considered ensemble members, pertains to <model> loop. For instance, ECMWF51 has 25 hindcast and 51 forecast members so <n_mem = 25> should be set if hindcasts and forecasts are combined in the skill evaluation. The first n_mem members are selected.
# n_lead = [8] #considered lead-time in months, pertains to <model> loop indicated with <mm> below, CMCC 3.5 hindcasts provide 183 forecast days, SEAS5.1 provide 215

# #set input parameters for model dataset to be aggregated, this example is for a single model
# variables = ['SPEI-3-R_eqm_pullLMs-TRUE','pvpot','SPEI-3-R','SPEI-3-M','fwi','psl','sfcWind','tas','pr','rsds'] #variable names in directories and file names
# variables_nc = ['SPEI-3-R','pvpot','SPEI-3-R','SPEI-3-M','FWI','psl','sfcWind','tas','pr','rsds'] #variable names within the netCDF files, differs in case of msi (msi is used within the file, but psl is used in the file name)
# variables_new = ['SPEI-3-R-C1','pvpot','SPEI-3-R','SPEI-3-M','fwi','msl','si10','t2m','tp','ssrd'] #new variable names; as provided by ERA5 data from CDS
# time_name = ['time','time','time','time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'] #name of the time dimension in the netCDF files for this variable, corresponds to <variables> input parameter and must have the same length
# lon_name = ['lon','lon','lon','lon','lon','x','x','x','x','x']
# lat_name = ['lat','lat','lat','lat','lat','y','y','y','y','y']
# file_start = ['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels_masked','seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'] #start string of the file names

# setup for various models
model = ['cmcc','ecmwf'] #seasonal forecast model
version = ['35','51'] #and version thereof; pertains to <model> loop indicated with <mm> below
n_mem = [40,25] #number of considered ensemble members, pertains to <model> loop. For instance, ECMWF51 has 25 hindcast and 51 forecast members so <n_mem = 25> should be set if hindcasts and forecasts are combined in the skill evaluation. The first n_mem members are selected.
n_lead = [7,8] #considered lead-time in months, pertains to <model> loop indicated with <mm> below, CMCC 3.5 hindcasts provide 183 forecast days, SEAS5.1 provide 215

# variables = [['pr','tas','psl','sfcWind','rsds'],['pr','tas','psl','sfcWind','rsds']] #variable names in directories and file names
# variables_nc = [['pr','tas','psl','sfcWind','rsds'],['pr','tas','psl','sfcWind','rsds']] #variable names within the netCDF files, differs in case of fwi msi (msi / FWI is used within the file, but psl / fwi is used in the file name)
# variables_new = [['tp','t2m','msl','si10','ssrd'],['tp','t2m','msl','si10','ssrd']] #new variable names; as provided by ERA5 data from CDS
# time_name = [['forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'],['forecast_time','forecast_time','forecast_time','forecast_time','forecast_time']] #list containing lists, each sublist specifies the names of the time coordinates for each variables of a specific model; one list per model
# lon_name = [['x','x','x','x','x'],['x','x','x','x','x']] #as time_name but for the name of the longitude coordinate
# lat_name = [['y','y','y','y','y'],['y','y','y','y','y']] #as time_name but for the name of the latitude coordinate
# file_start = [['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels']] ##as time_name but for the name of the head folder containing hindcasts and predictions

variables = [['SPEI-3-M','pr','tas','psl','sfcWind','rsds'],['SPEI-3-M','pr','tas','psl','sfcWind','rsds']] #variable names in directories and file names
variables_nc = [['SPEI-3-M','pr','tas','psl','sfcWind','rsds'],['SPEI-3-M','pr','tas','psl','sfcWind','rsds']] #variable names within the netCDF files, differs in case of fwi msi (msi / FWI is used within the file, but psl / fwi is used in the file name)
variables_new = [['SPEI-3-M','tp','t2m','msl','si10','ssrd'],['SPEI-3-M','tp','t2m','msl','si10','ssrd']] #new variable names; as provided by ERA5 data from CDS
time_name = [['time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time'],['time','forecast_time','forecast_time','forecast_time','forecast_time','forecast_time']] #list containing lists, each sublist specifies the names of the time coordinates for each variables of a specific model; one list per model
lon_name = [['lon','x','x','x','x','x'],['lon','x','x','x','x','x']] #as time_name but for the name of the longitude coordinate
lat_name = [['lat','y','y','y','y','y'],['lat','y','y','y','y','y']] #as time_name but for the name of the latitude coordinate
file_start = [['seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels'],['seasonal-original-single-levels_masked','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels','seasonal-original-single-levels']] ##as time_name but for the name of the head folder containing hindcasts and predictions

years = [[1993,2023],[1981,2023]] #years to be regridded, the output files will be filled with monthly values, aggregated from daily values in this script, covering all months beginning in January of the indicated start year and ending in December of the indicated end year. If no daily input data is found for a given month, nans will be placed in the monthly output netCDF files.

#fixed parameters not passing through loops
imonth = [1,2,3,4,5,6,7,8,9,10,11,12] #month the forecasts are initialized on, 1 refers to the January 1st, 2 to Febrary 1st etc.

#set MEDCOF domain
domain = 'medcof' #spatial domain the model data is available on. So far, this is just a label used to find the input files and name the output files.
save_corrected_files = 'no' #overwrite input nc files with un expected units with newly generated corrected files

#setup for path to the input GCM files
gcm_store = 'lustre' #laptop, F, lustre or extdisk2

## EXECUTE #############################################################
if save_corrected_files == 'yes':
    print('WARNING: the <save_corrected_files> input parameter was set to '+save_corrected_files+' by the user and input files with unexpected units will be corrected and OVER-WRITTEN ! Be sure that this is intended !')
xr.set_options(file_cache_maxsize=1)

#set path to input gcm files as a function of the file system set by the <gcm_store> input parameter
if gcm_store == 'laptop':
    home = os.getenv('HOME')
    path_gcm_base = home+'/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    savepath_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/gcm/aggregated' #Directory of the output files generated by this script
elif gcm_store == 'F':
    home = os.getenv('HOME')
    path_gcm_base = '/media/swen/F/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    savepath_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/gcm/aggregated' #Directory of the output files generated by this script
elif gcm_store == 'extdisk2':
    home = os.getenv('HOME')
    path_gcm_base = '/media/swen/ext_disk2/datos/GCMData/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = path_gcm_base # head directory of the source files
    path_gcm_base_masked = path_gcm_base # head directory of the source files
    savepath_base = home+'/datos/tareas/proyectos/pticlima/seasonal/results/gcm/aggregated' #Directory of the output files generated by this script
elif gcm_store == 'lustre':
    home = '/lustre/gmeteo/PTICLIMA'
    path_gcm_base = home+'/DATA/SEASONAL/seasonal-original-single-levels' # head directory of the source files
    path_gcm_base_derived = home+'/DATA/SEASONAL/seasonal-original-single-levels_derived' # head directory of the source files
    path_gcm_base_masked = home+'/DATA/SEASONAL/seasonal-original-single-levels_masked' # head directory of the source files
    savepath_base = home+'/Results/seasonal/gcm/aggregated' #Directory of the output files generated by this script
else:
    raise Exception('ERROR: unknown entry for <path_gcm_base> !')
print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

#get mesh information from static file, assumed to be equal for all models
nc_template = xr.open_dataset(path_gcm_base+'/'+domain+'/hindcast/tas/ecmwf/51/198101/seasonal-original-single-levels_medcof_hindcast_tas_ecmwf_51_198101.nc')
n_lat = len(nc_template['y'])
n_lon = len(nc_template['x'])
nc_template.close()

for mm in np.arange(len(model)):
    #create directory of the output netcdf files if necessary
    if os.path.isdir(savepath_base+'/'+model[mm]+version[mm]) != True:
        os.makedirs(savepath_base+'/'+model[mm]+version[mm])
    print('INFO: aggregating '+model[mm]+version[mm]+' data for '+str(n_mem[mm])+' '+str(years[mm][0])+' to '+str(years[mm][-1])+' and months '+str(imonth).replace('[','').replace(',','').replace(']',''))
    
    years_vec = np.arange(years[mm][0],years[mm][1]+1) #create an array of year
    n_mon = len(years_vec)*12+n_lead[mm]-1 #length of the monthly time series in the output data array generated by this script, 12 is hard-coded because it refers to the number of months per year and therefore is a static variable, if the initialization is not provided in <imonth> the output data is set to nan

    for vv in np.arange(len(variables[mm])):
        #init the numpy array to be filled with this script; one array per variable and model that will be transferred to xarray format and saved to netCDF
        data_mon = np.zeros((n_mon,n_lead[mm],n_mem[mm],n_lat,n_lon))
        data_mon[:] = np.nan
        datelist = [] #init date list
        
        #construct path to input GCM files as a function of the variable set in variables[mm][vv]
        if variables[mm][vv] in ('fwi','pvpot'):
            path_gcm_base_var = path_gcm_base_derived
        elif variables[mm][vv] in ('SPEI-3','SPEI-3-M','SPEI-3-R','SPEI-3-R_eqm_pullLMs-TRUE'):
            path_gcm_base_var = path_gcm_base_masked
        elif variables[mm][vv] in ('psl','sfcWind','tas','pr','rsds'):
            path_gcm_base_var = path_gcm_base
        else:
            raise Exception('ERROR: check entry for variables[mm][vv] !')
            
        for yy in np.arange(len(years_vec)):
            #Check whether to use hindcasts or forecasts
            if years_vec[yy] > 2016 and model[mm]+version[mm] in ('ecmwf51','cmcc35'):
                print('Info: No hindcast available for '+model[mm]+version[mm]+' and year '+str(years_vec[yy])+'. The forecast is loaded instead...')
                product = 'forecast'
            else:
                #print('Info: Loading hindcast for '+model[mm]+version[mm]+' and year '+str(years_vec[yy])+'...')
                product = 'hindcast'                
            #load each monthly init separately, aggregate daily to monthly-mean data and fill the numpy array <data_mon> interatively.
            for im in np.arange(len(imonth)):
                print('INFO: Loading '+variables[mm][vv]+' from '+model[mm]+version[mm]+' on '+domain+' domain for '+str(imonth[im]).zfill(2)+' '+str(years_vec[yy]))
                
                #print complete path to the GCM input file
                if variables[mm][vv] in ('SPEI-3','SPEI-3-M','SPEI-3-R'):
                    #path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/coefs_all_members/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/coefs_pool_members/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                elif variables[mm][vv] in ('SPEI-3-R_eqm_pullLMs-TRUE'):
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/coefs_of_reanalysis/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_SPEI-3-R_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                elif variables[mm][vv] in ('fwi','pvpot'):
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                elif variables[mm][vv] in ('psl','sfcWind','tas','pr','rsds'):
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                else:
                    raise Exception('ERROR: check entry for variables[mm][vv] !')
                
                #check whether netCDF file for years_vec[yy] and imonth[im], containing the monthly gcm intis, exists. This is done because a try because some files are missing. If the file is not there, continue to the next step of the loop / month indexed by <im>. 
                if os.path.isfile(path_gcm_data):
                    nc = xr.open_dataset(path_gcm_data)
                else:
                    print('WARNING: '+path_gcm_data+' is not available! Proceed to load the next netCDF file containing '+model[mm]+version[mm]+' data for '+str(years_vec[yy])+' and '+str(imonth[im]))
                    continue
                    
                #check if the latitudes are in the right order or must be flipped to be consistent with the obserations used for validation
                if nc[lat_name[mm][vv]][0].values < nc[lat_name[mm][vv]][-1].values:
                    print('WARNING: the latitudes in '+path_gcm_data+' come in ascending order and are inverted to be consistent with the order of the remaining datasets / variables (descending) !')
                    if lat_name[mm][vv] == 'lat':
                        nc = nc.reindex(lat=list(reversed(nc.lat)))
                    elif lat_name[mm][vv] == 'y':
                        nc = nc.reindex(y=list(reversed(nc.y)))
                    else:
                        raise Exception('ERROR: unexpected entry for <lat_name[mm][vv]> !')
                elif nc[lat_name[mm][vv]][0].values > nc[lat_name[mm][vv]][-1].values:
                    print('INFO: the latitudes in '+path_gcm_data+' come in descending order, as excpected to be consistent with the order of the remaining datasets.')
                elif nc[[mm][vv]][0].values == nc[lat_name[mm][vv]][-1].values:
                    print('WARNING = the southern and northernmost latitude values are the same in '+path_gcm_data+' !')
                else:
                    raise Exception('ERROR: unexpected latitude order in '+path_gcm_data)
                
                #transform GCM variables and units, if necessary
                nc, file_valid = transform_gcm_variable(nc,variables_nc[mm][vv],variables_new[mm][vv],model[mm],version[mm])
                
                ##select ensemble members. This is done because the ensemble members in the forecast period may be more than in the hindcast period, e.g. 25 vs. 50 in ECWMF 51
                print('INFO: Selecting the first '+str(n_mem[mm])+' ensemble members from a total of '+str(nc.member.shape[0])+' members...') 
                nc = nc.isel(member=np.arange(n_mem[mm]))
                
                nc_mon = nc[variables_nc[mm][vv]].resample(time="1MS").mean(dim=time_name[mm][vv]) #https://stackoverflow.com/questions/50564459/using-xarray-to-make-monthly-average
                dates_mon = pd.DatetimeIndex(nc_mon.time)
                datelist = datelist+list(nc_mon.time.values) #concatenate the date list to create a monthly unique date vector covering the whole period considered in <years>; see below.
                if yy == 0 and im == 0:
                    # data_mon = np.zeros((n_mon,n_lead[mm],len(nc_mon['member']),nc_mon.shape[2],nc_mon.shape[3]))
                    # data_mon[:] = np.nan

                    ##get dimensions and metadata to save in output netCDF file below
                    try:
                        var_units = nc[variables_nc[mm][vv]].units
                    except:
                        print('WARNING: no units have been stored in netcdf input file for '+variables[mm][vv]+' located at '+path_gcm_data)
                        if variables_nc[mm][vv] == 'pvpot':
                            print('Units are set to: zero-bound index')
                            var_units = 'zero-bound index'
                        else:
                            raise Exception('ERROR: no units are defined for '+variables[mm][vv]+' !!')
                        
                    #var_name = nc[variables_nc[mm][vv]].name #optionally use variable name from input netCDF files
                    
                    ##get members from the input file and transform the variety of possible formats to integer
                    #members = nc.member.values.astype(int) #this option does not work for the first SPEI-3-R version
                    members = np.array([int(str(nc.member[ii].astype(str).values).replace('Member_','')) for ii in np.arange(len(nc.member))]) #this option also works for the first SPEI-3-R version
                    members = np.arange(len(members)) #force the members to start with 0 irrespective of the input format (the first SPEI-3-R version started with 1)
                    
                    #try to retrieve region defintion (i.e. the domain) from the input netCDF file. The region is provided by the files from Predictia but not so by fwi files. If not provided, the value in the <domain> input parameter is set.
                    try:
                        region = nc.region.values
                    except:
                        print('WARNING: The region / domain is not provided with the input file '+path_gcm_data+'. Consequently, it is set to '+domain+', this values being provided by the <domain> input parameter set by the user above in this script.')
                        region = domain
                        
                    lons = nc[lon_name[mm][vv]].values
                    #lon_attrs = nc[lon_name[mm][vv]].attrs #currently not used
                    lats = nc[lat_name[mm][vv]].values
                    #lat_attrs = nc[lat_name[mm][vv]].attrs #currently not used
                
                pos_time = dates_mon.month-1 + ((dates_mon.year - years_vec[0])* 12)
                pos_lead = np.arange(len(dates_mon.month)) #corresponding position along the "lead" dimension
                data_mon[pos_time,pos_lead,:,:,:] = nc_mon.values
                nc.close() #close the currently open nc file object
                
                #if the input netCDF file is not valid, as revealed by transform_gcm_variable(), overwrite the original file with the newly created one
                if (file_valid == 0) & (save_corrected_files == 'yes'):
                    del(nc) #delete the previously opened nc file object
                    nc = xr.open_dataset(path_gcm_data) #and re-open it
                    #get encoding and variable unit of the input netCDF file for all dimensions and variables
                    ec_x = nc[lon_name[mm][vv]].encoding
                    ec_y = nc[lat_name[mm][vv]].encoding
                    ec_member = nc['member'].encoding
                    ec_forecast_reference_time = nc['forecast_reference_time'].encoding
                    ec_region = nc['region'].encoding
                    ec_time = nc['time'].encoding
                    ec_forecast_time = nc['forecast_time'].encoding
                    ec_var = nc[variables_nc[vv]].encoding
                    units_var = nc[variables_nc[vv]].units
                    
                    nc[variables_nc[mm][vv]].values = (nc[variables_nc[mm][vv]].astype(ec_var.get('dtype'))+273.15).values #set to float64 to be coherente with input data and subtract 273.15 to transform Kelvin to degrees Celsius and thus be coherent with the other ecmwf 51 input file download and processed by Predictia
                    #nc[variables_nc[mm][vv]].attrs['units'] = units_var, is not necessary is previous lines works with .values; otherwise, the variable attributes are dropped and shoud be re-assigned here
                    path_corrected_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/corrected_seasonal-original-single-levels_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                    print('INFO: The units of the input array were wrong and have been corrected. The corrected array with units coherent to the remaining files is newly stored in '+path_corrected_gcm_data+' !')
                    #encoding = {'x': ec_x, 'y': ec_x, 'member': ec_member, 'forecast_reference_time': ec_forecast_reference_time, 'region': ec_region, 'time': ec_time, 'forecast_time': ec_forecast_time, variables_nc[mm][vv]: ec_var}
                    # Valid encodings are: {'chunksizes', 'least_significant_digit', 'shuffle', 'complevel', 'compression', 'fletcher32', 'dtype', 'zlib', '_FillValue', 'contiguous'}
                    #encoding = {variables_nc[mm][vv]: {'dtype': ec_var.get('dtype'), 'zlib': ec_var.get('zlib'), 'complevel': ec_var.get('complevel'), 'chunksizes': ec_var.get('chunksizes'), '_FillValue': ec_var.get('_FillValue'), 'shuffle': ec_var.get('shuffle'), 'fletcher32': ec_var.get('fletcher32'), 'contiguous': ec_var.get('contiguos')}}
                    var_encoding = {variables_nc[mm][vv]: {'dtype': ec_var.get('dtype'), 'zlib': ec_var.get('zlib'), 'complevel': ec_var.get('complevel'), '_FillValue': ec_var.get('_FillValue'), 'shuffle': ec_var.get('shuffle'), 'fletcher32': ec_var.get('fletcher32'), 'contiguous': ec_var.get('contiguos')}}
                    nc.to_netcdf(path_corrected_gcm_data,encoding=var_encoding)
                    nc.close()
                    del(nc)
                    
                    #replace the old erroneous file with the new corrected one
                    print('INFO: Upon user request the erroneous file '+path_gcm_data+' is replaced with the new corrected file '+path_corrected_gcm_data+' !')
                    os.rename(path_corrected_gcm_data,path_gcm_data)
                    
                elif (file_valid == 0) & (save_corrected_files == 'no'):
                    print('INFO: The units of the input array were wrong and have been corrected for further processing with the pySeasonal package. However, as requested by the user, the wrong input netCDF file is not overwritten with the corrected file !')
                    nc.close()
                    del(nc)
                elif file_valid == 1:
                    print('INFO: The input file is valid following the criteria defined in transform_gcm_variable() and no corrections are necessary.')
                else:
                    nc.close()
                    del(nc)
                    raise Exception('ERROR: unknown entry for <file_valid> !')
                
                nc_mon.close()
                del(nc_mon)
                print(pos_time)
        
        ## generate pandas Datetime object with all unique dates, create xarray DataArray and save to netCDF
        datelist = pd.DatetimeIndex(datelist).unique()
        startdate = str(datelist.year[0])+'-01-01 00:00:00' #the start calendar day of the output nc file is fixed at 1st of January for ECMWF
        enddate = str(years[mm][-1]+1)+'-'+str(n_lead[mm]-1).zfill(2)+startdate[-12:] #the end calendar day of the output nc file is fixed at 1st of July for ECMWF51 due to the 8-month leadtime (from December 1 of the previous year to July 1 of the end year)
        leads = np.arange(n_lead[mm])
        daterange = pd.date_range(start=startdate, end=enddate, freq='MS') #MS = month start frequency, see https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        outnc = xr.DataArray(data_mon, coords=[daterange, leads, members, lats, lons], dims=['time', 'lead', 'member', 'y', 'x'], name=variables_new[mm][vv])
        del(data_mon) #delete the loop-wise numpy array <data_mon>

        ##cut out the last n_lead[mm]-1 months to harmonize the time dimension with observations, currently not used because this is done afterwards in <get_skill.py> 
        #yearbool = (daterange.year >= years[0]) & (daterange.year <= years[1])
        #outnc = outnc.isel(time = yearbool)
        
        ##set netCDF attributes
        #variable attributes
        outnc.attrs['units'] = var_units
        outnc.attrs['standard_name'] = variables_new[mm][vv]
        outnc.attrs['long_name'] = variables_new[mm][vv]        
        ##time attributes
        #outnc.time.attrs['units'] = 'month' #The script runs into an error if this attribute is set because outnc cannot be saved
        outnc.time.attrs['standard_name'] = 'time'
        outnc.time.attrs['long_name'] = 'time'
        outnc.time.attrs['description'] = 'The forecast is valid for the indicated month and year.'
        #lead attribtues
        outnc.lead.attrs['units'] = 'month'
        outnc.lead.attrs['standard_name'] = 'lead'
        outnc.lead.attrs['long_name'] = 'forecast lead-time'
        outnc.lead.attrs['description'] = 'The forecast lead-time in months, i.e. 0 = from the init date to the last day of the first month, 1 = from the first day to the last day of the second month, and so on.'
        #member attributes
        outnc.member.attrs['units'] = 'categorical, '+str(members[0])+' to '+str(members[-1])
        outnc.member.attrs['standard_name'] = 'member'
        outnc.member.attrs['long_name'] = 'ensemble member'
        outnc.member.attrs['description'] = 'indicator of model member within the '+model[mm]+version[mm]+' ensemble as provided by the input files'
        #lon and lat attributes
        outnc.x.attrs['units'] = 'degrees_east'
        outnc.x.attrs['standard_name'] = 'longitude'
        outnc.x.attrs['long_name'] = 'longitude'
        outnc.y.attrs['units'] = 'degrees_north'
        outnc.y.attrs['standard_name'] = 'latitude'
        outnc.y.attrs['long_name'] = 'latitude'
        
        savename = savepath_base+'/'+model[mm]+version[mm]+'/'+variables_new[mm][vv]+'_mon_'+model[mm]+version[mm]+'_'+str(n_mem[mm])+'m_'+domain+'_'+str(years[mm][0])+'_'+str(years[mm][-1])+'.nc'
        outnc.to_netcdf(savename)
        outnc.close()

print('INFO: aggregate_hindcast.py has been run successfully! The output nc files containing the temporally aggregeted GCM data can be found in '+savepath_base)

