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
import sys
import yaml
from pathlib import Path

# INDICATE CONFIGURATION FILE ######################################

configuration_file = 'config_for_aggregate_hindcast_Canarias.yaml'
# configuration_file = 'config_for_aggregate_hindcast_Canarias.yaml'

####################################################################

#this is a function to load the configuration file
def load_config(config_file='config/'+configuration_file):
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / config_file
    print('The path of the configuration file is '+str(config_path))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Setup paths based on GCM_STORE environment variable
    gcm_store = os.getenv('GCM_STORE', 'lustre')
    if gcm_store in config['path']:
        paths = config['path'][gcm_store]
        config['paths'] = paths
    else:
        raise ValueError('Unknown entry for <gcm_store> !')
    
    return config

# Load configuration
config = load_config()

# Extract configuration variables
model = config['model']
version = config['version']
n_mem = config['n_mem']
n_lead = config['n_lead']

# The following variables are lists of lists, with the outer list corresponding to the models.
# We construct them from the model_settings in the config file.
model_keys = [m + v for m, v in zip(model, version)]
variables = [config['model_settings'][key]['variables'] for key in model_keys]
variables_nc = [config['model_settings'][key]['variables_nc'] for key in model_keys]
variables_new = [config['model_settings'][key]['variables_new'] for key in model_keys]
time_name = [config['model_settings'][key]['time_name'] for key in model_keys]
lon_name = [config['model_settings'][key]['lon_name'] for key in model_keys]
lat_name = [config['model_settings'][key]['lat_name'] for key in model_keys]
file_start = [config['model_settings'][key]['file_start'] for key in model_keys]
years = [config['model_settings'][key]['years'] for key in model_keys]

# details example file per model that is available on your file system; the example file must be for the base variables as provided by Jaime, i.e. no climate indices are permitted
template_init = config['template_init']
template_var = config['template_var']
template_file_start = config['template_file_start']
template_lon = config['template_lon']
template_lat = config['template_lat']

#fixed parameters not passing through loops
imonth = config['imonth']

#set MEDCOF domain
domain = config['domain']
save_corrected_files = config['save_corrected_files']

# Extract paths from configuration
paths = config['paths'] # get paths from configuration
home = paths['home']
rundir = paths['rundir']
path_gcm_base = paths['path_gcm_base']
path_gcm_base_derived = paths['path_gcm_base_derived']
path_gcm_base_masked = paths['path_gcm_base_masked']
savepath_base = paths['savepath_base']
FLAGDIR = paths['FLAGDIR']

## EXECUTE #############################################################
os.chdir(rundir) #go to running directory

#load config file and custom functions
exec(open('functions_seasonal.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

#exec(open(rundir+'/config/config_for_aggregate_hindcast.py').read()) #reads the <functions_seasonal.py> script containing a set of custom functions needed here

# #alternatively, add directories to your system path in oder to import the configuration files and functions; this however requires restarting Pyhton every time the functions and config files are updated, which is why the exec option mentioned above is preferred for script development
# sys.path.append(rundir) #add the running directory
# sys.path.append(rundir+'/config') #add the dirctory containg the config files
# print('IMPORTANT NOTE: The configuration file containing all input parameters is located at: '+rundir+'/config')

# #import all input variables located in the config directory, go to running directory, and load the functions script located there
# from config_for_aggregate_hindcast import model, version, n_mem, n_lead, variables, variables_nc, variables_new, time_name, lon_name, lat_name, file_start, years, imonth, domain, save_corrected_files
# from functions_seasonal import * #import all my custom functions

#create output directory needed by this script, if needed.
if os.path.isdir(FLAGDIR) != True:
    os.makedirs(FLAGDIR)

if save_corrected_files == 'yes':
    print('WARNING: the <save_corrected_files> input parameter was set to '+save_corrected_files+' by the user and input files with unexpected units will be corrected and OVER-WRITTEN ! Be sure that this is intended !')
xr.set_options(file_cache_maxsize=1)

print('The GCM files will be loaded from the base directory '+path_gcm_base+'...')

#get mesh information from static file, assumed to be equal for all models
nc_template = xr.open_dataset(path_gcm_base+'/'+domain+'/hindcast/'+template_var[0]+'/ecmwf/51/198101/'+template_file_start[0]+'_'+domain+'_hindcast_'+template_var[0]+'_ecmwf_51_198101.nc', decode_timedelta=False)

n_lat = len(nc_template[template_lat[0]])
n_lon = len(nc_template[template_lon[0]])
nc_template.close()

for mm in np.arange(len(model)):

    #get model dimensions from a template modell initialization, needed for entirely missing hindcast months. This init data of this template file and other information is set in <config_for_aggregate_hindcast.yaml>; <members>, <lons> and <lats> from this file will be overwritten if other init files are found during execution of this script
    if template_var[mm] in ('tas','psl','pr','FD-C4','SU-C4','TR-C4'):
        path_template_gcm = path_gcm_base+'/'+domain+'/hindcast/'+template_var[mm]+'/'+model[mm]+'/'+version[mm]+'/'+str(template_init[mm][0:4])+str(template_init[mm][-2:]).zfill(2)+'/'+template_file_start[mm]+'_'+domain+'_hindcast_'+template_var[mm]+'_'+model[mm]+'_'+version[mm]+'_'+str(template_init[mm][0:4])+str(template_init[mm][-2:]).zfill(2)+'.nc'
        nc_template_gcm = xr.open_dataset(path_template_gcm, decode_timedelta=False)
        nc_template_gcm = nc_template_gcm.isel(member=np.arange(n_mem[mm])) #select members within the template netCDF file
        members = np.array([int(str(nc_template_gcm.member[ii].astype(str).values).replace('Member_','')) for ii in np.arange(len(nc_template_gcm.member))]) #this option also works for the first SPEI-3-R version
        members = np.arange(len(members)) #force the members to start with 0 irrespective of the input format (the first SPEI-3-R version started with 1)
        lons = nc_template_gcm[template_lon].values
        lats = nc_template_gcm[template_lat].values
        # #make a short latitute check
        # if lats[0] < lats[-1]:
        #     raise ValueError('latitudes in <lats> must be descending !')
    else:
        raise ValueError('<template_var[mm]> is not in the list of allowed template variables defined in config_for_aggregate_hindcast.yaml ! This is because the template variables must have a specific format, e.g. ascending latitudes, that is, e.g., not met by the SPEI-3 inidices')

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
        elif variables[mm][vv] in ('SPEI-3','SPEI-3-M','SPEI-3-R','SPEI-3-R_eqm_pullLMs-TRUE','FD-C4','SU-C4','TR-C4'):
            path_gcm_base_var = path_gcm_base_masked
        elif variables[mm][vv] in ('psl','sfcWind','tas','pr','rsds'):
            path_gcm_base_var = path_gcm_base
        else:
            raise ValueError('Check entry for variables[mm][vv] !')
            
        for yy in np.arange(len(years_vec)):
            #Check whether to use hindcasts or forecasts
            if years_vec[yy] > 2016 and model[mm]+version[mm] in ('ecmwf51','cmcc35','cmcc4'):
                print('Info: No hindcast available for '+model[mm]+version[mm]+' and year '+str(years_vec[yy])+'. The forecast is loaded instead...')
                product = 'forecast'
            elif years_vec[yy] > 2023 and model[mm]+version[mm] in ('eccc5','dwd22'):
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
                elif variables[mm][vv] in ('fwi','pvpot','FD-C4','SU-C4','TR-C4'):
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                elif variables[mm][vv] in ('psl','sfcWind','tas','pr','rsds'):
                    path_gcm_data = path_gcm_base_var+'/'+domain+'/'+product+'/'+variables[mm][vv]+'/'+model[mm]+'/'+version[mm]+'/'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'/'+file_start[mm][vv]+'_'+domain+'_'+product+'_'+variables[mm][vv]+'_'+model[mm]+'_'+version[mm]+'_'+str(years_vec[yy])+str(imonth[im]).zfill(2)+'.nc'
                else:
                    raise Exception('ERROR: check entry for variables[mm][vv] !')
                
                #check whether netCDF file for years_vec[yy] and imonth[im], containing the monthly gcm intis, exists. This is done because a try because some files are missing. If the file is not there, continue to the next step of the loop / month indexed by <im>. 
                if os.path.isfile(path_gcm_data):
                    nc = xr.open_dataset(path_gcm_data, decode_timedelta=False)
                else:
                    print('WARNING: '+path_gcm_data+' is not available! Proceed to load the next netCDF file containing '+model[mm]+version[mm]+' data for '+str(years_vec[yy])+' and '+str(imonth[im]))
                    var_units = 'not known because no file is available'
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
                            print('For '+variables_nc[mm][vv]+', the units are set to: zero-bound index')
                            var_units = 'zero-bound index'
                        # if variables_nc[mm][vv] in ('FD','SU','TR'):
                        #     print('For '+variables_nc[mm][vv]+', the units are set to: number of days')
                        #     var_units = 'number of days'
                        else:
                            raise ValueError('No units are defined for '+variables_nc[mm][vv]+' !!')
                        
                    #var_name = nc[variables_nc[mm][vv]].name #optionally use variable name from input netCDF files
                    
                    ##get members from the input file and transform the variety of possible formats to integer
                    #members = nc.member.values.astype(int) #this option does not work for the first SPEI-3-R version
                    
                    #this overwrites <members> from the template per model file <nc_template_gcm> loaded above
                    del(members)
                    members = np.array([int(str(nc.member[ii].astype(str).values).replace('Member_','')) for ii in np.arange(len(nc.member))]) #this option also works for the first SPEI-3-R version
                    members = np.arange(len(members)) #force the members to start with 0 irrespective of the input format (the first SPEI-3-R version started with 1)
                    
                    #try to retrieve region defintion (i.e. the domain) from the input netCDF file. The region is provided by the files from Predictia but not so by fwi files. If not provided, the value in the <domain> input parameter is set.
                    try:
                        region = nc.region.values
                    except:
                        print('WARNING: The region / domain is not provided with the input file '+path_gcm_data+'. Consequently, it is set to '+domain+', this values being provided by the <domain> input parameter set by the user above in this script.')
                        region = domain

                    #this overwrites <lons and lats> from the template per model file <nc_template_gcm> loaded above
                    del(lons,lats)  
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
                    nc = xr.open_dataset(path_gcm_data, decode_timedelta=False) #and re-open it
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
        
        if np.all(np.isnan(data_mon)):
            pdb.set_trace()
            
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

#write flag documenting the correct execution of this script
print('INFO: aggregate_hindcast.py has been run successfully. A flag is written at '+FLAGDIR)
flagfile = FLAGDIR+'/aggregate_hindcast_'+domain+'_corrfiles_'+save_corrected_files+'_'+str(model)+'_newvars_'+str(variables_new)+'.flag'
flagfile = flagfile.replace("[","").replace("]","").replace("'","").replace(",","_").replace(" ","")
file = open(flagfile,'w')
file.write('aggregate_hindcast.py has been run successfully for '+save_corrected_files+', '+str(model)+', '+str(version)+', '+str(variables)+', '+str(variables_nc)+', '+str(variables_new))
file.close()

print('INFO: aggregate_hindcast.py has been run successfully! The output nc files containing the temporally aggregeted GCM data can be found in '+savepath_base)

