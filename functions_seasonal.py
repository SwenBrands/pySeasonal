#!/usr/bin/env python

'''functions used in PTI clima'''

from math import radians, cos, sin, asin, sqrt
import numpy as np

def apply_sea_mask(arr_f,mask_file_f):
    '''sets the values over the Sea, as provided by the netCDF file locted at <mask_file_f> to nan in <arr_f>;
    Input: <arr_f> is an xarray DataArray with the dimensions detrended, variable, time, season, lead, y, x;
    <mask_file_f> is a character string pointing to the path of the maks file in netCDF format. The function tests
    whether the dimensions in <arr_f> are as expected and whether the latitutes in <arr_f> therein match those in <mask_file_f>.
    Ouput: returns the modified <arr_f> with nan values over the Sea'''
    
    nc_mask_f = xr.open_dataset(mask_file_f) #open the mask file

    #test for equal latitudes
    if ~np.all(nc_mask_f.latitude.values-arr_f.y.values == 0):
        ValueError('<nc_mask_f.latitude> and <arr_f.y> do not match !')

    # target_dims = list(dict(arr_f.dims).values())
    
    # test if dimensions in <arr_f> are as expected
    target_dims = arr_f.dims
    if target_dims != ('detrended', 'variable', 'time', 'season', 'lead', 'y', 'x'):
        ValueError('The dimensions of <arr_f> are not as expected !')
    
    mask_appended_f = np.tile(nc_mask_f.mask.values,(arr_f.shape[0],arr_f.shape[1],arr_f.shape[2],arr_f.shape[3],arr_f.shape[4],1,1))
    arr_f = arr_f.where(~np.isnan(mask_appended_f), np.nan) #retain grid-boxes marked with 1 in mask

    #clean everything except arr_f, which will be returned to the script calling this function
    nc_mask_f.close()
    del(nc_mask_f,mask_appended_f)
    return(arr_f) #returns masked xarray data array

def assign_season_label(season_list_f):
    '''assign the season string for a the input list of 3 consecutive months, each month being an integer.'''
    # 1-months seasons
    if season_list_f == [1]:
        season_label_f = 'JAN'
    elif season_list_f == [2]:
        season_label_f = 'FEB'
    elif season_list_f == [3]:
        season_label_f = 'MAR'
    elif season_list_f == [4]:
        season_label_f = 'APR'
    elif season_list_f == [5]:
        season_label_f = 'MAY'
    elif season_list_f == [6]:
        season_label_f = 'JUN'
    elif season_list_f == [7]:
        season_label_f = 'JUL'
    elif season_list_f == [8]:
        season_label_f = 'AUG'
    elif season_list_f == [9]:
        season_label_f = 'SEP'
    elif season_list_f == [10]:
        season_label_f = 'OCT'
    elif season_list_f == [11]:
        season_label_f = 'NOV'
    elif season_list_f == [12]:
        season_label_f = 'DEC'
    # 2-months seasons
    elif season_list_f == [1,2]:
        season_label_f = 'JF'
    elif season_list_f == [2,3]:
        season_label_f = 'FM'
    elif season_list_f == [3,4]:
        season_label_f = 'MA'
    elif season_list_f == [4,5]:
        season_label_f = 'AM'
    elif season_list_f == [5,6]:
        season_label_f = 'MJ'
    elif season_list_f == [6,7]:
        season_label_f = 'JJ'
    elif season_list_f == [7,8]:
        season_label_f = 'JA'
    elif season_list_f == [8,9]:
        season_label_f = 'AS'
    elif season_list_f == [9,10]:
        season_label_f = 'SO'
    elif season_list_f == [10,11]:
        season_label_f = 'ON'
    elif season_list_f == [11,12]:
        season_label_f = 'ND'
    elif season_list_f == [12,1,]:
        season_label_f = 'DJ'
    # 3-months seasons
    elif season_list_f == [1,2,3]:
        season_label_f = 'JFM'
    elif season_list_f == [2,3,4]:
        season_label_f = 'FMA'
    elif season_list_f == [3,4,5]:
        season_label_f = 'MAM'
    elif season_list_f == [4,5,6]:
        season_label_f = 'AMJ'
    elif season_list_f == [5,6,7]:
        season_label_f = 'MJJ'
    elif season_list_f == [6,7,8]:
        season_label_f = 'JJA'
    elif season_list_f == [7,8,9]:
        season_label_f = 'JAS'
    elif season_list_f == [8,9,10]:
        season_label_f = 'ASO'
    elif season_list_f == [9,10,11]:
        season_label_f = 'SON'
    elif season_list_f == [10,11,12]:
        season_label_f = 'OND'
    elif season_list_f == [11,12,1]:
        season_label_f = 'NDJ'
    elif season_list_f == [12,1,2]:
        season_label_f = 'DJF'
    # 4-months seasons
    elif season_list_f == [1,2,3,4]:
        season_label_f = 'JFMA'
    elif season_list_f == [2,3,4,5]:
        season_label_f = 'FMAM'
    elif season_list_f == [3,4,5,6]:
        season_label_f = 'MAMJ'
    elif season_list_f == [4,5,6,7]:
        season_label_f = 'AMJJ'
    elif season_list_f == [5,6,7,8]:
        season_label_f = 'MJJA'
    elif season_list_f == [6,7,8,9]:
        season_label_f = 'JJAS'
    elif season_list_f == [7,8,9,10]:
        season_label_f = 'JASO'
    elif season_list_f == [8,9,10,11]:
        season_label_f = 'ASON'
    elif season_list_f == [9,10,11,12]:
        season_label_f = 'SOND'
    elif season_list_f == [10,11,12,1]:
        season_label_f = 'ONDJ'
    elif season_list_f == [11,12,1,2]:
        season_label_f = 'NDJF'
    elif season_list_f == [12,1,2,3]:
        season_label_f = 'DJFM'
    # 5-months seasons
    elif season_list_f == [1,2,3,4,5]:
        season_label_f = 'JFMAM'
    elif season_list_f == [2,3,4,5,6]:
        season_label_f = 'FMAMJ'
    elif season_list_f == [3,4,5,6,7]:
        season_label_f = 'MAMJJ'
    elif season_list_f == [4,5,6,7,8]:
        season_label_f = 'AMJJA'
    elif season_list_f == [5,6,7,8,9]:
        season_label_f = 'MJJAS'
    elif season_list_f == [6,7,8,9,10]:
        season_label_f = 'JJASO'
    elif season_list_f == [7,8,9,10,11]:
        season_label_f = 'JASON'
    elif season_list_f == [8,9,10,11,12]:
        season_label_f = 'ASOND'
    elif season_list_f == [9,10,11,12,1]:
        season_label_f = 'SONDJ'
    elif season_list_f == [10,11,12,1,2]:
        season_label_f = 'ONDJF'
    elif season_list_f == [11,12,1,2,3]:
        season_label_f = 'NDJFM'
    elif season_list_f == [12,1,2,3,4]:
        season_label_f = 'DJFMA'
    else:
        raise Exception('ERROR: check entry for <season_list_f> !')
    return(season_label_f)

# def get_forecast_prob(nc_quantile,seas_mean):
def get_forecast_prob(seas_mean,lower_xr,upper_xr):
    '''Obtains probability forecasts for each tercile in <seas_mean>, using the quantiles stored in <nc_quantile>; all 3 are xarray data arrays;
    seas_mean is 3D with time x lat x lon, lower_xr and upper_xr are 2d with lat x lon'''

    lower_np = np.tile(lower_xr.values,(seas_mean.shape[0],1,1))
    upper_np = np.tile(upper_xr.values,(seas_mean.shape[0],1,1))
    
    upper_ind = (seas_mean > upper_np) & (~np.isnan(upper_np))
    center_ind = (seas_mean >= lower_np) & (seas_mean <= upper_np) & (~np.isnan(upper_np))
    lower_ind = (seas_mean < lower_np) & (~np.isnan(lower_np))
                    
    #sum members in each category and devide by the number of members, thus obtaining the probability
    nr_mem = upper_ind.shape[0]
    upper_prob = upper_ind.sum(dim='member')/nr_mem
    center_prob = center_ind.sum(dim='member')/nr_mem
    lower_prob = lower_ind.sum(dim='member')/nr_mem
    return(nr_mem,upper_prob,center_prob,lower_prob)

def get_years_of_subperiod(subperiod_f):
    ''' obtain target years used for validation as a function of the sole input parameter <subperiod_f>.
      ENSO years were derived from CPC's ONI index available from https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
      or from the NOAA list available at https://psl.noaa.gov/enso/past_events.html 
      QBO years were derived from CPCs QBO index at 50mb available at https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index'''
    if subperiod_f == 'mod2strong_nino_oni':    
        years_val = [1982,1983,1986,1987,1991,1992,1997,1998,2009,2010,2015,2016] #Swen Brands selection based no NOAA's ONI index
        print('The model is verified for moderate and strong El Niño years based on ONI index only: '+str(years_val))
    elif subperiod_f == 'mod2strong_nina_oni':
        years_val = [1984,1985,1988,1989,1999,2000,2007,2008,2010,2011,2020,2021,2022]
        print('The model is verified for moderate and strong La Niña years based on ONI index only: '+str(years_val))
    elif subperiod_f == 'enso_nino_noaa':    
        years_val = [1983,1987,1988,1992,1995,1998,2003,2007,2010,2016]
        print('The model is verified for the El Niño years declared by NOAA at: https://psl.noaa.gov/enso/past_events.html : '+str(years_val))
    elif subperiod_f == 'enso_nina_noaa':    
        years_val = [1989,1999,2000,2008,2011,2012,2021,2022]
        print('The model is verified for the La Niña years declared by NOAA at: https://psl.noaa.gov/enso/past_events.html : '+str(years_val))
    elif subperiod_f == 'enso_neutral_noaa':    
        years_val = [1981,1982,1984,1985,1986,1990,1991,1993,1994,1996,1997,2001,2002,2004,2005,2006,2009,2013,2014,2015,2017,2018,2019,2020] #Swen Brands selection based no NOAA's ONI index
        print('The model is verified for the neutral ENSO years declared by NOAA at: https://psl.noaa.gov/enso/past_events.html : '+str(years_val))
    elif subperiod_f == 'qbo50_pos':
        years_val = [1981,1983,1985,1986,1988,1991,1993,1995,1997,1999,2000,2002,2004,2009,2011,2014,2017,2019,2021,2023]
        print('The model is verified for positive QBO-50 years derived from https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: '+str(years_val))
    elif subperiod_f == 'qbo50_neg':
        years_val = [1982,1984,1987,1989,1992,1994,1996,1998,2001,2003,2005,2007,2010,2012,2015,2018,2022]
        print('The model is verified for negative QBO-50 years derivded from https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: '+str(years_val))
    elif subperiod_f == 'qbo50_trans':
        years_val = [1990,2006,2008,2013,2016,2020]
        print('The model is verified for transition QBO-50 years derived from https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: '+str(years_val))
    elif subperiod_f == 'none':
        years_val = np.arange(1981,2023,1)
        print('The full overlapping period between observations and model data is used for verification.')
    else:
        raise Exception('ERROR: unkown entry for the <subperiod> entry parameter !')
    return(years_val)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def roll_and_cut(xr_dataset, lonlim_f, latlim_f):
    """
    rolls a global gridded regular lat-lon dataset from 0 <= lon < 360 to -180 <= lon < 180 format and cuts out a specific region
    input: xr_dataset: xarray dataset; lonlim_f and latlim_f = two lists containing floats, these are the longitude and latitude limits to be cut out
    """
    #first, roll the dataset so that -180 < lon <= 180
    whemis = np.where(xr_dataset.longitude > 180)[0]
    newlons = xr_dataset.longitude.values
    newlons[whemis] = newlons[whemis]-360
    xr_dataset['longitude'] = newlons
    shiftat = int(np.argmax(np.abs(np.diff(xr_dataset.longitude.values)))-1)
    xr_dataset = xr_dataset.roll(longitude=shiftat,roll_coords=True)
    
    #then cut out target region and return new dataset
    lonind = (xr_dataset.longitude.values >= lonlim_f[0]) & (xr_dataset.longitude.values <= lonlim_f[1])
    latind = (xr_dataset.latitude.values >= latlim_f[0]) & (xr_dataset.latitude.values <= latlim_f[1])
    xr_dataset = xr_dataset.isel(longitude=lonind,latitude=latind)
    return(xr_dataset)

def calc_roll_seasmean(xr_ds):
    """
    calculates rolling weighted seasonal average values considering DJF, MAM, JJA and SON
    input: xarray dataset containing a temporal sequence of monthly mean values covering all calenar months, e.g. 01-1980, 02-1981, 03-1981, 04-1981 etc.
    """
    month_length = xr_ds.time.dt.days_in_month
    weights = month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    #np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4)) # Test that the sum of the weights for each season is 1.0
    xr_ds_roll = xr_ds*weights
    xr_ds_roll = xr_ds_roll.rolling(time=3,center=False,min_periods=None).sum()
    roll_weights = weights.rolling(time=3,center=False,min_periods=None).sum()
    xr_ds_roll = xr_ds_roll / roll_weights
    id3 = np.arange(0,xr_ds_roll.time.shape[0],3) # index used to retain every third value of the rolling 3-months seasonal mean values to obtain inter-annual time-series.
    xr_ds_roll = xr_ds_roll.isel(time=id3)
    return(xr_ds_roll)
    xr_ds_roll.close()
    weights.close()
    del(weights,xr_ds_roll)

def lin_detrend(xr_ar,rm_mean_f):
    """also used in pyLamb package; performs linear detrending of the xarray DataArray xr_ar along the time dimension, rm_mean_f specifies whether the mean is removed yes or no"""

    #raise an error if time is not the first dimension in xr_ar
    if  xr_ar.dims.index('time') != 0:
        ValueError('The first dimension in the xarray data array xr_ar must be "time" !')

    coeff = xr_ar.polyfit(dim='time',deg=1,skipna=True) #deg = 1 for linear detrending
    fit = xr.polyval(xr_ar['time'], coeff.polyfit_coefficients)
    if rm_mean_f == 'yes':
        xr_ar_detrended = xr_ar - fit
    elif rm_mean_f == 'no':
        tiles_f = np.ones(len(xr_ar.dims))
        tiles_f[0] = len(xr_ar.time)
        meanvals_f = np.tile(xr_ar.mean(dim='time'),tiles_f.astype('int'))
        xr_ar_detrended = xr_ar - fit + meanvals_f
    else:
        raise Exception('ERROR: check entry for <rm_mean_f> input parameter!')
    return(xr_ar_detrended)

#def get_frac_significance(np_arr_pval_f,np_arr_rho_f,critval_f,mode_f='fraction',lat_f=None):
#def get_spatial_aggregation(np_arr_pval_f,np_arr_rho_f,critval_f,mode_f='fraction',lat_f=None):
def get_spatial_aggregation(score_f,critval_f=None,pval_f=None,mode_f='fraction_pos',lat_f=None):
    """get the fraction of grid-boxes where significant results are obtained in percentage of all grid-boxes forming the domain; score_f and pval_f a 4d numpy array
    with the dimensions season x lead x lat x lon"""
    
    shape_f = score_f.shape
    score_step_f = np.reshape(score_f,[shape_f[0],shape_f[1],shape_f[2]*shape_f[3]])
    #find grid-boxes set to nan, e.g. because they are over sea
    nanind_rho_f = np.where(np.isnan(score_step_f[0,0,:]))[0]
    #remove nan grid-boxes
    score_step_f = np.delete(score_step_f,nanind_rho_f,axis=2)

    if pval_f is not None:
        pval_step_f = np.reshape(pval_f,[shape_f[0],shape_f[1],shape_f[2]*shape_f[3]])
        nanind_pval_f = np.where(np.isnan(pval_step_f[0,0,:]))[0]    
        #check whether the two nan indices representing nan grid-boxes are identical
        if np.any(nanind_rho_f != nanind_pval_f) == True:
            raise Exception('ERROR in get_frac_significance() function ! The indices <nanind_rho_f> and <nanind_pval_f> do not match !')
        #remove nan gridboxes
        pval_step_f = np.delete(pval_step_f,nanind_rho_f,axis=2)
    
    #if lat_f is passed in anything but None, then latitudinal weights are calculated and adapted to the format of <score_step_f>
    if lat_f is not None:
        lat_weights = np.cos(np.radians(np.reshape(lat_f,lat_f.shape[0]*lat_f.shape[1])))
        lat_weights = np.delete(lat_weights,nanind_rho_f,axis=0)
        lat_weights = np.tile(lat_weights,[score_step_f.shape[0],score_step_f.shape[1],1])
    
    if mode_f in ('fraction_smaller','fraction_larger','fraction_smaller_pos'): #caclulate the areal percentage of significant correlation coefficients
        if mode_f == 'fraction_smaller_pos':
            sigind_f = (pval_step_f < critval_f) & (score_step_f > 0) 
            spurind_f = (pval_step_f >= critval_f) | (score_step_f <= 0)
            pval_step_f[sigind_f] = 1
            pval_step_f[spurind_f] = 0
        elif mode_f == 'fraction_smaller': #caclulate the areal percentage of significant correlation coefficients
            sigind_f = pval_step_f < critval_f 
            spurind_f = pval_step_f >= critval_f
            pval_step_f[sigind_f] = 1
            pval_step_f[spurind_f] = 0 
        elif mode_f == 'fraction_larger': #caclulate the areal percentage of significant correlation coefficients
            sigind_f = pval_step_f > critval_f 
            spurind_f = pval_step_f <= critval_f
            pval_step_f[sigind_f] = 1
            pval_step_f[spurind_f] = 0
        else:
            raise Excpetion('ERROR: The <mode_f> parameter is unknown !')
        if lat_f is not None:
            print('As requested by the user, the latitude-weighted areal-mean value is calculated...')
            pval_step_f = pval_step_f*lat_weights
            spatial_agg_f = (np.nansum(pval_step_f,axis=2)/np.nansum(lat_weights,axis=2))*100
        else:
            print('As requested by the user, the simple areal-mean value is calculated...')
            spatial_agg_f = (np.nansum(pval_step_f,axis=2)/score_step_f.shape[2])*100
    elif mode_f == 'mean': #calculate the areal-mean correlation coefficient
        if lat_f is not None:
            print('As requested by the user, the latitude-weighted areal-mean value is calculated...')
            score_step_f = score_step_f*lat_weights
            spatial_agg_f = np.nansum(score_step_f,axis=2)/np.nansum(lat_weights,axis=2)
        else:
            print('As requested by the user, the simple areal-mean value is calculated...')
            spatial_agg_f = np.nansum(score_step_f,axis=2)/(score_step_f.shape[2])
    else:
        raise Exception('ERROR: check entry for <mode_f> parameter within the get_frac_significance() function !')
    return(spatial_agg_f) #the spatially aggregated value is returned
    #return(spatial_sigfraq_f,pval_step_f) #former versions of this function returned two output variables

def get_frac_above_threshold(np_arr_vals_f,critval_f):
    """get the fraction of grid-boxes where values i values_f exceed the threshold <critval_f>; np_arr_f is a 4d numpy array with the dimensions season x lead x lat x lon"""
    shape_f = np_arr_vals_f.shape
    np_arr_vals_step_f = np.reshape(np_arr_vals_f,[shape_f[0],shape_f[1],shape_f[2]*shape_f[3]])
    sigind_f = np_arr_vals_step_f > critval_f 
    spurind_f = np_arr_vals_step_f <= critval_f
    np_arr_vals_step_f[sigind_f] = 1
    np_arr_vals_step_f[spurind_f] = 0
    spatial_fraq_f = np.nansum(np_arr_vals_step_f,axis=2)/(shape_f[2]*shape_f[3])*100
    return(spatial_fraq_f,np_arr_vals_step_f)

def get_sub_domain(xr_ds_f,domain_f):
    '''cuts out the sub-domain defined in <domain_f> from xarray dataset <xr_ds_f>'''
    #check whether the requested sub-domain is known; otherwise return an error
    if sub_domain == 'iberia':
        print('Upon user request, verification results for '+domain_f+' will be cut out.')
        lat_bool = (xr_ds_f.y.values >= 36) & (xr_ds_f.y.values <= 44)
        lon_bool = (xr_ds_f.x.values >= -10) & (xr_ds_f.x.values <= 3)
        xr_ds_f = xr_ds_f.isel(y=lat_bool,x=lon_bool)
        # #set grid-boxes in North Africa to nan
        # latind_f = xr_ds_f.y.values <= 37
        # lonind_f = xr_ds_f.x.values >= -1
        # xr_ds_f.loc[dict(y=latind_f, x=lonind_f)] = np.nan
    elif sub_domain == 'medcof2': #this domain is identical to the medcof domain, but does not include the Sahara desert. The SPEI does not cover this area.
        lat_bool = (xr_ds_f.y.values >= 28) & (xr_ds_f.y.values <= 90)
        lon_bool = (xr_ds_f.x.values >= -16) & (xr_ds_f.x.values <= 180)
        xr_ds_f = xr_ds_f.isel(y=lat_bool,x=lon_bool)
    else:
        raise Exception('ERROR: check entry for the <sub_domain> input parameter !')
    return(xr_ds_f)
    xr_ds_f.close()
    del(xr_ds_f)

def plot_pcolormesh_seasonal(xr_ar_f,minval_f,maxval_f,savename_f,colormap_f,dpival_f):
    '''Plots matrix of the verfication results contained in xarray data array <xr_ar_f>, seasons are plotted on the x axis, lead months on the y axis.'''
    fig = plt.figure()
    ax = xr_ar_f.plot.pcolormesh(cmap = colormap_f, x = 'season', y = 'lead', vmin = minval_f, vmax = maxval_f, add_colorbar=False, rasterized=True)
    ax.axes.set_yticks(xr_ar_f.lead.values)
    ax.axes.set_yticklabels(xr_ar_f.lead_label,fontsize=8)
    ax.axes.set_xticks(xr_ar_f.season.values)
    ax.axes.set_xticklabels(xr_ar_f.season_label,fontsize=8,rotation=45.)
    ax.axes.set_aspect('auto')
    #plt.xticks(fontsize=5)
    #plt.xlabel(xr_ar_f.season.name)
    #plt.ylabel(xr_ar_f.lead.name)
    plt.xlabel('Month the forecast valid for')
    plt.ylabel('Lead time (months)')
    cbar = plt.colorbar(ax,shrink=0.5,label=xr_ar_f.name + ' ('+xr_ar_f.units+')', orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=8,size=8)
    fig.tight_layout()
    plt.savefig(savename_f,dpi=dpival_f)
    plt.close('all')

def get_map_lowfreq_var(pattern_f,xx_f,yy_f,agree_ind_f,minval_f,maxval_f,dpival_f,title_f,savename_f,halfres_f,colormap_f,titlesize_f,cbarlabel_f,origpoint=None,orientation_f=None):
    '''Currently used in pyLamb and pySeasonal packages in sligthly differing versions. Plots a pcolormesh contour over a map overlain by dots indicating, e.g. statistical significance'''
    
    map_proj = ccrs.PlateCarree()
    
    fig = plt.figure()
    toplayer_x = xx.flatten()[agree_ind_f.flatten()]
    toplayer_y = yy.flatten()[agree_ind_f.flatten()]
    maxind = np.argsort(pattern_f.flatten())[-1]
    max_x = xx_f.flatten()[maxind]
    max_y = yy_f.flatten()[maxind]
    minind = np.argsort(pattern_f.flatten())[0]
    min_x = xx_f.flatten()[minind]
    min_y = yy_f.flatten()[minind]

    ax = fig.add_subplot(111, projection=map_proj)
    ax.set_extent([xx_f.min()-halfres, xx_f.max()+halfres_f, yy.min()-halfres_f, yy_f.max()+halfres_f], ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE, zorder=4, color='black')
            
    image = ax.pcolormesh(xx_f, yy_f, pattern_f, vmin=minval_f, vmax=maxval_f, cmap=colormap_f, transform=ccrs.PlateCarree(), shading = 'nearest', zorder=3, rasterized=True)
    #get size of the points indicating significance
    if halfres_f < 1.:
        pointsize_f = 0.25
        marker_f = '+'
    else:
        pointsize_f = 0.5
        marker_f = 'o'
    #plot the top layer which may indicate any kind of indication, e.g. statistical significance
    if agree_ind_f.sum() > 0:
        ax.plot(toplayer_x, toplayer_y, color='black', marker=marker_f, linestyle='none', markersize=pointsize_f, transform=ccrs.PlateCarree(), zorder=4)
    #plot a single point on the map, e.g. a city location
    if origpoint != None:
        ax.plot(origpoint[0], origpoint[1], color='blue', marker='X', linestyle='none', markersize=2, transform=ccrs.PlateCarree(), zorder=5)
    ##plot parallels and meridians
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=0.5, linestyle='dotted', zorder=6)
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    
    #set orientation of the colorbar
    if orientation_f is None:
        orientation_f = 'vertical'
        rotation_f = 270
    else:
        rotation_f = 0
    cbar = plt.colorbar(image,orientation=orientation_f, pad=0.03, shrink = 0.6) #pad defines the distance between the panel and its colormap
    cbar.set_label(cbarlabel_f, rotation=rotation_f, labelpad=4, y=0.5, fontsize=titlesize_f) #labelpad defines the distance between the colorbar and its label
    cbar.ax.tick_params(labelsize=titlesize_f)
    plt.title(title_f, fontsize=titlesize_f-1)
    plt.savefig(savename_f,dpi=dpival_f,bbox_inches='tight')
    plt.close('all')   

def transform_gcm_variable(ds_f,var_in_f,var_out_f,model_f,version_f):
    '''transforms GCM variable names and units to be compatible with CDS nomenclature; input: <ds_f> is an xarray dataset, <var_in_f> is the name
    of the input meteorological variable, <var_out_f> is the new name (or output name) of this variable; <model_f> and <version_f> are the name
     and version of the modelling system; output: xarray dataset <ds_f> with corrected variable names and units.'''

    # go through exceptions depending on the variable, model, version, etc.    
    if (var_in_f == 'tas') & (model_f+version_f in ('ecmwf51','cmcc35','cmcc4','eccc5')):
        #bring temperature data to Kelvin, taking into account Predictia's double transformation error in all forecasts from 201701 to 202311
        if (ds_f[var_in_f].mean().values <= 100) & (ds_f[var_in_f].mean().values > -100):
            print('Info: Adding 273.15 to '+var_in_f+' data from '+model_f+version_f+' to transform degress Celsius into Kelvin.')
            ds_f[var_in_f].values = ds_f[var_in_f].values+273.15
            valid_f = 1
        elif ds_f[var_in_f].mean().values <= -100:
            #raise Exception('ERROR: the '+var_in_f+' values starting at '+str(ds_f.time[0].values)+' are too low !')
            print('Info: The file has been transformed twice! Adding 2x273.15 to '+var_in_f+' data from '+model_f+version_f+' to correct Predictia workflow error and then transform into Kelvin.')
            ds_f[var_in_f][:] = ds_f[var_in_f]+273.15+273.15
            valid_f = 0
        else:
            raise Exception('ERROR: Unknown value for <ds_f[var_in_f]> !')
        ds_f[var_in_f].attrs['units'] = 'daily mean '+var_out_f+' in Kelvin'
    elif (var_in_f in ('pr','rsds')) & (model_f+version_f in ('ecmwf51','cmcc35','cmcc4','eccc5')):
        print('Info: Disaggregate '+var_in_f+' accumulated over the '+str(len(ds_f.time))+' days forecast period from '+model_f+version_f+' to daily sums.')
        vals_disagg = np.diff(ds_f[var_in_f].values,n=1,axis=0)
        vals_disagg[vals_disagg < 0] = 0 #set negative flux values to 0
        shape_disagg = vals_disagg.shape
        add_day = np.expand_dims(vals_disagg[-1,:,:,:],axis=0) #get last available difference to place on the last day
        #add_day = np.zeros((1,shape_disagg[1],shape_disagg[2],shape_disagg[3])) #optionally create a nan to be placed on the last day
        #add_day[:] = np.nan
        vals_disagg = np.concatenate((vals_disagg,add_day),axis=0) #add last available difference to the remaining data to avoid a nan
        ds_f[var_in_f][:] = vals_disagg
        ds_f[var_in_f].attrs['units'] = 'daily accumulated '+var_out_f+' in '+ds_f[var_in_f].attrs['units']
        valid_f = 1
    else:
        print('Info: No data transformation is applied for '+var_in_f+' data from '+model_f+version_f+'.')
        valid_f = 1
    return(ds_f, valid_f)

def get_reliability_or_roc(obs_f,gcm_f,obs_quantile_f,gcm_quantile_f,dist_part_f,score_f='reliability',bin_edges_f=None):
    """ caclulates the mean absolute difference (returned as <reliability>) between the forecast probabilities and corresponding conditional observed probabilities of the reliability plot and the diagonal of the plot
    obs_f (5d in get_skill_season.py) and gcm_f (6d) are xarray data arrays, obs_quantile_f and gcm_quantile_f are pre-caclulated observed and gcm quantiles in xarray format,
    dist_part_f is a character string indicating the part of the distribution that will be assessed; currently possible values are "lower_tercile", "center_tercile" and "upper_tercile"
    """
    
    if len(obs_f.dims) == 4:
        ## obtain observed quantile thresholds and replicate along the time axis to fit the shape of obs_f
        obs_lower_tercile_f = np.tile(obs_quantile_f.sel(quantile=1/3),(obs_f.shape[0],1,1,1))
        obs_upper_tercile_f = np.tile(obs_quantile_f.sel(quantile=2/3),(obs_f.shape[0],1,1,1))    
        #replicate the gcm quantile thresholds passed to this function via <gcm_quantile_f> along the time axis
        gcm_lower_tercile_f = np.tile(gcm_quantile_f.sel(quantile=1/3),(gcm_f.shape[0],1,1,1,1)) #get target threshold value from precalculated threshold values and replicate along the time dimension to fit the size of gcm_f
        gcm_upper_tercile_f = np.tile(gcm_quantile_f.sel(quantile=2/3),(gcm_f.shape[0],1,1,1,1))
    elif len(obs_f.dims) == 5:
        ## obtain observed quantile thresholds and replicate along the time axis to fit the shape of obs_f
        obs_lower_tercile_f = np.tile(obs_quantile_f.sel(quantile=1/3),(obs_f.shape[0],1,1,1,1))
        obs_upper_tercile_f = np.tile(obs_quantile_f.sel(quantile=2/3),(obs_f.shape[0],1,1,1,1))    
        #replicate the gcm quantile thresholds passed to this function via <gcm_quantile_f> along the time axis
        gcm_lower_tercile_f = np.tile(gcm_quantile_f.sel(quantile=1/3),(gcm_f.shape[0],1,1,1,1,1)) #get target threshold value from precalculated threshold values and replicate along the time dimension to fit the size of gcm_f
        gcm_upper_tercile_f = np.tile(gcm_quantile_f.sel(quantile=2/3),(gcm_f.shape[0],1,1,1,1,1))
    else:
        raise Exception('Error in get_reliability_or_roc(): check the number of dimenions in <obs_f>!')  

    #get binary time series, observed or modelled time series are within the part of the distribution specified in <dist_part> yes (1) or no (0)  
    if dist_part_f == 'upper_tercile':
        obs_bin = xr.where(obs_f > obs_upper_tercile_f, 1, 0).astype('int8') #here the nan values over the sea are lost. They will be brought back below.
        gcm_bin = xr.where(gcm_f > gcm_upper_tercile_f, 1, 0).astype('int8')
    elif dist_part_f == 'lower_tercile':
        obs_bin = xr.where(obs_f < obs_lower_tercile_f, 1, 0).astype('int8') #here the nan values over the sea are lost. They will be brought back below.
        gcm_bin = xr.where(gcm_f < gcm_lower_tercile_f, 1, 0).astype('int8')
    elif dist_part_f in ('center_tercile','centre_tercile'):
        obs_bin = xr.where((obs_f >= obs_lower_tercile_f) & (obs_f <= obs_upper_tercile_f), 1, 0).astype('int8') #here the nan values over the sea are lost. They will be brought back below.
        gcm_bin = xr.where((gcm_f >= gcm_lower_tercile_f) & (gcm_f <= gcm_upper_tercile_f), 1, 0).astype('int8')
    else:
        raise Exception("ERROR: check entry for dist_part_f !")
        
    # obs_bin = obs_bin.where(~np.isnan(obs_f)) #xs.reliability and xs.roc do not work with nans in the input arrays, so this line is commented so far
    # gcm_bin = gcm_bin.where(~np.isnan(gcm_f)) #xs.reliability and xs.roc do not work with nans in the input arrays, so this line is commented so far
    
    # #manually remove the first time instant so far, as long as xskill does not treat nans for this function
    # if pd.DatetimeIndex(gcm_f.time).year[0] == 1981:
    #     print('WARNING: the first year in the observed (obs_bin) and forecasted occurrence / absence (gcm_bin) time series is 1981 and is removed by the get_reliability() function because of nans present in <gcm_bin> during the first year of evaluation that cannot be handled by xskillscore.reliability() so far.') 
    #     obs_bin = obs_bin.isel(time=slice(1, None))
    #     gcm_bin = gcm_bin.isel(time=slice(1, None))
    
    #caclulate the score indicated in the <score_f> input parameter
    if score_f == 'reliability': #calculate reliability as defined in Wilks (2006)
        print('As requested by the user, the RELIABILITY is calculated by the get_reliability_or_roc() function.')
        if bin_edges_f is None: #use default number of bins
            o_cond_y = xs.reliability(obs_bin, gcm_bin.mean("member"), dim='time').drop('samples') #see Wilks 2006, returns the observed relative frequencies (o) conditional to 5 (= default values) forecast probability bins y (0.1, 0.3, 0.5, 0.7, 0.9), see https://xskillscore.readthedocs.io/en/stable/api/xskillscore.reliability.html#xskillscore.reliability 
        else: #use bins whose edges are provided by the optional <bin_edges_f> input parameter
            o_cond_y = xs.reliability(obs_bin, gcm_bin.mean("member"), dim='time',probability_bin_edges=bin_edges_f).drop('samples')
        
        #process as a function of the number of dimension in obs_f, gcm_f, obs_quantile_f and gcm_quantile_f
        if len(obs_f.dims) == 4:
            diagonal = np.tile(o_cond_y.forecast_probability.values,(o_cond_y.shape[0],o_cond_y.shape[1],o_cond_y.shape[2],1)) #this is the diagonal of the reliability diagramm
            reliability = np.abs(o_cond_y - diagonal).mean(dim='forecast_probability') #calculate the residual (i.e. absolute difference) from the diagonal averged over the 5 forecast bins mentioned above
            #reliability = reliability.where(~np.isnan(obs_f[0,:,:,:])) # re-set the grid-boxes over sea to nan as in the input data values
        elif len(obs_f.dims) == 5:
            diagonal = np.tile(o_cond_y.forecast_probability.values,(o_cond_y.shape[0],o_cond_y.shape[1],o_cond_y.shape[2],o_cond_y.shape[3],1)) #this is the diagonal of the reliability diagramm
            reliability = np.abs(o_cond_y - diagonal).mean(dim='forecast_probability') #calculate the residual (i.e. absolute difference) from the diagonal averged over the 5 forecast bins mentioned above
            #reliability = reliability.where(~np.isnan(obs_f[0,:,:,:,:])) # re-set the grid-boxes over sea to nan as in the input data values
        else:
            raise Exception('Error in get_reliability_or_roc(): check the number of dimenions in obs_f, gcm_f, obs_quantile_f and gcm_quantile_f !')
        out_score = reliability
    elif score_f == 'roc_auc': #caclulate roc area under the curve
        print('As requested by the user, the ROC area under the curve is calculated by the get_reliability_or_roc() function.')
        roc = xs.roc(obs_bin, gcm_bin.mean("member"), dim='time',  bin_edges='continuous', drop_intermediate=False, return_results='area')
        out_score = roc
    else:
        raise Exception('ERROR: unknown entry for <score_f> input parameter in get_reliability_or_roc() function !')
    
    if len(obs_f.dims) == 4:
        out_score = out_score.where(~np.isnan(obs_f[1,:,:,:])) # re-set the grid-boxes over sea to nan as in the input data values, index must be set to 1 because index = 0 coincides with nan for DJF and NDJ seasons !!
    elif len(obs_f.dims) == 5:
        out_score = out_score.where(~np.isnan(obs_f[1,:,:,:,])) # re-set the grid-boxes over sea to nan as in the input data values, index must be set to 1 because index = 0 coincides with nan for DJF and NDJ seasons !!
    else:
        raise Exception('Error in get_reliability_or_roc(): check the number of dimenions in <obs_f>!')
    
    return(out_score)
