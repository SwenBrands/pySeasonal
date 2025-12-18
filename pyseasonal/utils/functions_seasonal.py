#!/usr/bin/env python

'''functions used in PTI clima'''

from math import radians, cos, sin, asin, sqrt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs

from pyseasonal.utils.mapping import subperiod_years

def flip_latitudes_and_data(xr_ds_f,lat_name):
    ''' flips latitudes and data variables in xr_ds_f; input: xr_df_f is an xarray dataseet and lat_name a string containing the placeholder name for the latitude in that dataset;
    output: flipped xr_ds_f '''
    if lat_name == 'lat':
        xr_ds_f = xr_ds_f.reindex(lat=list(reversed(xr_ds_f.lat)))
    elif lat_name == 'y':
        xr_ds_f= xr_ds_f.reindex(y=list(reversed(xr_ds_f.y)))
    else:
        raise ValueError('Unexpected entry for <lat_name> in flip_latitudes() function !')

    return xr_ds_f


def apply_sea_mask(arr_f,mask_file_f,lat_name_f,lon_name_f):
    '''sets the values over the Sea, as provided by the netCDF file locted at <mask_file_f> to nan in <arr_f>;
    Input: <arr_f> is an xarray DataArray or Dataset with specific dimensions checked below;
    <mask_file_f> is a character string pointing to the path of the mask file in netCDF format; lat_name_f and lon_name_f are the latitude
    and longitude names inside <arr_f>; the lat name in the mask file MUST be "lat" ! The function tests performs various consistency tests before actually applying the mask.
    Ouput: returns the modified <arr_f> with nan values over the Sea'''

    nc_mask_f = xr.open_dataset(mask_file_f) #open the mask file

    #check whether the latitudes in the mask file are descinding; otherwise return error
    if nc_mask_f.lat[0] <= nc_mask_f.lat[-1]:
        raise ValueError(' The latitudes in '+mask_file_f+' are ASCENDING and must be passed DESCENDING to apply_sea_mask() function ! ')
    elif nc_mask_f.lat[0] > nc_mask_f.lat[-1]:
        print(' The latitudes in '+mask_file_f+' are DESCENDING, as expected by the apply_sea_mask() function !')
    else:
        raise ValueError('unknown latitude order in '+mask_file_f)

    #check whether the y coordinate values in the xarray DataArray are are descinding; otherwise return error
    if arr_f[lat_name_f][0] <= arr_f[lat_name_f][-1]:
        raise ValueError(' The y coordinate values in <arr_f> are ASCENDING and must be passed DESCENDING to apply_sea_mask() function ! ')
    elif arr_f[lat_name_f][0] > arr_f[lat_name_f][-1]:
        print(' The y coordinate values in <arr_f> are DESCENDING, as expected by the apply_sea_mask() function !')
    else:
        raise ValueError('unknown y coordinate order in <arr_f> !')

    #test for equal latitudes
    if ~np.all(nc_mask_f.lat.values-arr_f[lat_name_f].values == 0):
        raise ValueError('<nc_mask_f.lat> and <arr_f.'+lat_name_f+' do not match !')

    # expand the mask, either for the xr DataArray passed via get_skill_season.py or for the xr Dataset passed via plot_seasonal_validation.results.py
    if isinstance(arr_f,xr.DataArray): #this is the format needed in get_skill_season.py
        print('<arr_f> in apply_sea_mask() function is an xarray DataArray !')
        # test if dimensions in <arr_f> are as expected
        target_dims_f = arr_f.dims
        if target_dims_f != ('detrended', 'variable', 'time', 'season', 'lead', lat_name_f, lon_name_f):
            raise ValueError('The dimensions of <arr_f> DataArray are not as expected !')
        elif target_dims_f == ('detrended', 'variable', 'time', 'season', 'lead', lat_name_f, lon_name_f):
            print('The dimensions of <arr_f> DataArray are as expected: '+str(target_dims_f))
        else:
            raise ValueError('Unknown values in <target_dims_f> within apply_sea_mask() function !')
        # extend the mask to match seven dimensions
        mask_appended_f = np.tile(nc_mask_f.mask.values,(arr_f.shape[0],arr_f.shape[1],arr_f.shape[2],arr_f.shape[3],arr_f.shape[4],1,1))

    elif isinstance(arr_f,xr.Dataset):#this is the format needed in plot_seasonal_validation_results.py
        print('<arr_f> in apply_sea_mask() function is an xarray Dataset !')
        first_arr_f = arr_f[list(arr_f.data_vars)[0]] #get first dataArray in dataset
        target_dims_f = first_arr_f.dims #get dimensions of the first data Variable in arr_f

        #check whether the coordinates sequence is as expected
        if target_dims_f == ('time', lat_name_f, lon_name_f): # for xr Datasets with 3 dimensions
            print('The dimensions of <first_arr_f> data array in <arr_f> dataset are as expected: '+str(target_dims_f))
            mask_appended_f = np.tile(nc_mask_f.mask.values,(first_arr_f.shape[0],1,1))
        elif target_dims_f == ('season', 'lead', lat_name_f, lon_name_f) or target_dims_f == ('time', 'member', lat_name_f, lon_name_f): # for xr Datasets with 4 dimensions
            print('The dimensions of <first_arr_f> data array in <arr_f> dataset are as expected: '+str(target_dims_f))
            mask_appended_f = np.tile(nc_mask_f.mask.values,(first_arr_f.shape[0],first_arr_f.shape[1],1,1))
        elif target_dims_f == ('time', 'lead', 'member', lat_name_f, lon_name_f): # # for xr Datasets with 5 dimensions
            print('The dimensions of <first_arr_f> data array in <arr_f> dataset are as expected: '+str(target_dims_f))
            mask_appended_f = np.tile(nc_mask_f.mask.values,(first_arr_f.shape[0],first_arr_f.shape[1],first_arr_f.shape[2],1,1))
        else:
            raise ValueError('Unknown values in <target_dims_f> within apply_sea_mask() function !')

        first_arr_f.close()
        del(first_arr_f)
    else:
        raise ValueError('unexpected instance for <arr_f> in apply_sea_mask() function !')

    arr_f = arr_f.where(~np.isnan(mask_appended_f), np.nan) #retain grid-boxes marked with 1 in mask

    #clean everything except arr_f, which will be returned to the script calling this function
    nc_mask_f.close()
    del(nc_mask_f,mask_appended_f,target_dims_f)

    return arr_f  #returns masked xarray data array


def assign_season_label(season_list_f: list) -> str:
    '''Assign season label string for an input list of consecutive months.

    Generates abbreviated season labels from month numbers. Handles single months
    (returns full month name like 'JAN') and multi-month seasons (returns initials
    like 'DJF' for December-January-February). Supports year wrap-around for seasons
    crossing December to January.

    Input:
        season_list_f: list of integers (1-12) representing consecutive months
                      Must contain 1-5 months in sequential order
                      Example: [12,1,2] for December-January-February

    Output:
        String label for the season
        Single month: full name (e.g., 'JAN', 'FEB', 'DEC')
        Multi-month: initials concatenated (e.g., 'DJF', 'JJA', 'MAMJ')

    Raises:
        ValueError: if months are not consecutive or list length is invalid
    '''
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    months_initials = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    # Check months validity
    if any(m not in range(1, 13) for m in season_list_f):
        raise ValueError(
            'ERROR: all month values in <season_list_f> must be integers between 1 and 12 !'
        )

    # Check list length
    if not (1 <= len(season_list_f) <= 5):
        raise ValueError(
            'ERROR: <season_list_f> must contain between 1 and 5 months only !'
        )

    if len(season_list_f) == 1:
        return months[season_list_f[0]-1]

    for i, m in enumerate(season_list_f[:-1]):
        if m == 12 and season_list_f[i+1] != 1:
            raise ValueError('ERROR: the months in <season_list_f> are not consecutive !')
        if m != 12 and season_list_f[i+1] - m != 1:
            raise ValueError('ERROR: the months in <season_list_f> are not consecutive !')

    return ''.join([months_initials[m-1] for m in season_list_f])


def get_forecast_prob(seas_mean_f,lower_xr_f,upper_xr_f):
    '''Obtains probability forecasts for each tercile in <seas_mean_f>, using the quantiles stored in <nc_quantile>; all 3 are xarray data arrays;
    seas_mean_f is 3D with time x lat x lon, lower_xr_f and upper_xr_f are 2d with lat x lon'''

    #check whether the NaN occurrence numbers in lower_xr_f and upper_xr_f are identical
    if np.sum(np.isnan(upper_xr_f)) - np.sum(np.isnan(lower_xr_f)) == 0:
        print(' The nan occurence number in both <upper_xr_f> and <lower_xr_f> is 0 within the get_forecast_prob() function ! ')
    elif np.sum(np.isnan(upper_xr_f)) - np.sum(np.isnan(lower_xr_f)) != 0:
        raise ValueError(' The nan occurence numbers in <upper_xr_f> and <lower_xr_f> do not match within the get_forecast_prob() function ! ')
    else:
        raise ValueError('Check entries for <upper_xr_f> and / or <lower_xr_f> in get_forecast_prob() function ! ')

    lower_np_f = np.tile(lower_xr_f.values,(seas_mean_f.shape[0],1,1))
    upper_np_f = np.tile(upper_xr_f.values,(seas_mean_f.shape[0],1,1))

    valid_ind_f = ~np.isnan(upper_np_f) & ~np.isnan(lower_np_f)
    upper_ind_f = (seas_mean_f > upper_np_f) & valid_ind_f
    center_ind_f = (seas_mean_f > lower_np_f) & (seas_mean_f <= upper_np_f) & valid_ind_f
    lower_ind_f = (seas_mean_f <= lower_np_f) & valid_ind_f

    #sum members in each category and devide by the number of members, thus obtaining the probability
    nr_mem_f = len(seas_mean_f.member)
    upper_prob_f = upper_ind_f.sum(dim='member')/nr_mem_f
    center_prob_f = center_ind_f.sum(dim='member')/nr_mem_f
    lower_prob_f = lower_ind_f.sum(dim='member')/nr_mem_f

    return nr_mem_f,upper_prob_f,center_prob_f,lower_prob_f


def get_years_of_subperiod(subperiod_f):
    ''' obtain target years used for validation as a function of the sole input parameter <subperiod_f>.
      ENSO years were derived from CPC's ONI index available from https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
      or from the NOAA list available at https://psl.noaa.gov/enso/past_events.html
      QBO years were derived from CPCs QBO index at 50mb available at https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index'''

    if subperiod_f not in subperiod_years:
        raise KeyError('ERROR: unknown entry for the <subperiod> entry parameter !')

    years_val = subperiod_years[subperiod_f]['years']
    print(subperiod_years[subperiod_f]['msg'].format(years=years_val))

    return years_val


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

    lonind = (xr_dataset.longitude.values >= lonlim_f[0]) & (xr_dataset.longitude.values <= lonlim_f[1])
    latind = (xr_dataset.latitude.values >= latlim_f[0]) & (xr_dataset.latitude.values <= latlim_f[1])
    xr_dataset = xr_dataset.isel(longitude=lonind,latitude=latind)

    return xr_dataset


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

    weights.close()
    del(weights)

    return xr_ds_roll


def lin_detrend(xr_ar,rm_mean_f):
    """also used in pyLamb package; performs linear detrending of the xarray DataArray xr_ar along the time dimension, rm_mean_f specifies whether the mean is removed yes or no"""

    #raise an error if time is not the first dimension in xr_ar
    if  xr_ar.dims.index('time') != 0:
        raise ValueError('The first dimension in the xarray data array xr_ar must be "time" !')

    coeff = xr_ar.polyfit(dim='time',deg=1,skipna=True).astype('float32') #deg = 1 for linear detrending
    fit = xr.polyval(xr_ar['time'], coeff.polyfit_coefficients).astype('float32')
    if rm_mean_f == 'yes':
        xr_ar_detrended = xr_ar - fit
    elif rm_mean_f == 'no':
        tiles_f = np.ones(len(xr_ar.dims))
        tiles_f[0] = len(xr_ar.time)
        meanvals_f = np.tile(xr_ar.mean(dim='time'),tiles_f.astype('int'))
        xr_ar_detrended = (xr_ar - fit + meanvals_f).astype('float32')
    else:
        raise Exception('ERROR: check entry for <rm_mean_f> input parameter!')

    return xr_ar_detrended


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
            raise Exception('ERROR: The <mode_f> parameter is unknown !')
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

    #return spatial_sigfraq_f,pval_step_f  #former versions of this function returned two output variables
    return spatial_agg_f  #the spatially aggregated value is returned


def get_frac_above_threshold(np_arr_vals_f,critval_f):
    """get the fraction of grid-boxes where values i values_f exceed the threshold <critval_f>; np_arr_f is a 4d numpy array with the dimensions season x lead x lat x lon"""
    shape_f = np_arr_vals_f.shape
    np_arr_vals_step_f = np.reshape(np_arr_vals_f,[shape_f[0],shape_f[1],shape_f[2]*shape_f[3]])
    sigind_f = np_arr_vals_step_f > critval_f
    spurind_f = np_arr_vals_step_f <= critval_f
    np_arr_vals_step_f[sigind_f] = 1
    np_arr_vals_step_f[spurind_f] = 0
    spatial_fraq_f = np.nansum(np_arr_vals_step_f,axis=2)/(shape_f[2]*shape_f[3])*100

    return spatial_fraq_f,np_arr_vals_step_f


def get_sub_domain(xr_ds_f,domain_f):
    '''cuts out the sub-domain defined in <domain_f> from xarray dataset <xr_ds_f>'''
    #check whether the requested sub-domain is known; otherwise return an error
    if domain_f == 'iberia':
        print('Upon user request, verification results for '+domain_f+' will be cut out.')
        lat_bool = (xr_ds_f.y.values >= 36) & (xr_ds_f.y.values <= 44)
        lon_bool = (xr_ds_f.x.values >= -10) & (xr_ds_f.x.values <= 3)
        xr_ds_f = xr_ds_f.isel(y=lat_bool,x=lon_bool)
        # #set grid-boxes in North Africa to nan
        # latind_f = xr_ds_f.y.values <= 37
        # lonind_f = xr_ds_f.x.values >= -1
        # xr_ds_f.loc[dict(y=latind_f, x=lonind_f)] = np.nan
    elif domain_f == 'medcof2': #this domain is identical to the medcof domain, but does not include the Sahara desert. The SPEI does not cover this area.
        lat_bool = (xr_ds_f.y.values >= 28) & (xr_ds_f.y.values <= 90)
        lon_bool = (xr_ds_f.x.values >= -16) & (xr_ds_f.x.values <= 180)
        xr_ds_f = xr_ds_f.isel(y=lat_bool,x=lon_bool)
    else:
        raise Exception('ERROR: check entry for the <sub_domain> input parameter !')

    return xr_ds_f


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
    ax.add_feature(cfeature.COASTLINE, zorder=4, color='black')

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
    if (var_in_f == 'tas') & (model_f+version_f in ('ecmwf51','cmcc35','cmcc4','eccc5','dwd22')):
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

    elif (var_in_f in ('pr','rsds')) & (model_f+version_f in ('ecmwf51','cmcc35','cmcc4','eccc5','dwd22')):
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

    return ds_f, valid_f


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
        obs_bin = xr.where(obs_f <= obs_lower_tercile_f, 1, 0).astype('int8') #here the nan values over the sea are lost. They will be brought back below.
        gcm_bin = xr.where(gcm_f <= gcm_lower_tercile_f, 1, 0).astype('int8')
    elif dist_part_f in ('center_tercile','centre_tercile'):
        obs_bin = xr.where((obs_f > obs_lower_tercile_f) & (obs_f <= obs_upper_tercile_f), 1, 0).astype('int8') #here the nan values over the sea are lost. They will be brought back below.
        gcm_bin = xr.where((gcm_f > gcm_lower_tercile_f) & (gcm_f <= gcm_upper_tercile_f), 1, 0).astype('int8')

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

    return out_score
