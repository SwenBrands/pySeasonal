#!/usr/bin/env python

'''functions used in PTI clima'''

from math import radians, cos, sin, asin, sqrt

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

def lin_detrend(xr_ar):
    """peforms linear detrending of the xarray DataArray xr_ar along the time dimension"""
    coeff = xr_ar.polyfit(dim='time',deg=1) #deg = 1 for linear detrending
    fit = xr.polyval(xr_ar['time'], coeff.polyfit_coefficients)
    xr_ar_detrended = xr_ar - fit
    return(xr_ar_detrended)

def get_fraq_significance(np_arr_f,critval_f):
    """get the fraction of grid-boxes where significant results are obtained in percentage of all grid-boxes forming the domain; np_arr_f is a 4d numpy array
    with the dimensions season x lead x lat x lon"""
    shape_f = np_arr_f.shape
    np_arr_step_f = np.reshape(np_arr_f,[shape_f[0],shape_f[1],shape_f[2]*shape_f[3]])
    sigind_f = np_arr_step_f < critval_f
    spurind_f = np_arr_step_f >= critval_f
    np_arr_step_f[sigind_f] = 1
    np_arr_step_f[spurind_f] = 0
    spatial_sigfraq_f = np.sum(np_arr_step_f,axis=2)/(shape_f[2]*shape_f[3])*100
    return(spatial_sigfraq_f)

def plot_pcolormesh_seasonal(xr_ar_f,minval_f,maxval_f,savename_f,colormap_f,dpival_f):
    '''Plots matrix of the verfication results contained in xarray data array <xr_ar_f>, seasons are plotted on the x axis, lead months on the y axis.'''
    fig = plt.figure()
    ax = xr_ar_f.plot.pcolormesh(cmap = colormap_f, x = 'season', y = 'lead', vmin = minval_f, vmax = maxval_f, add_colorbar=False)
    ax.axes.set_yticks(xr_ar_f.lead.values)
    ax.axes.set_yticklabels(xr_ar_f.lead.values,fontsize=4)
    ax.axes.set_xticks(xr_ar_f.season.values)
    ax.axes.set_xticklabels(xr_ar_f.season,fontsize=2, rotation = 45.)
    ax.axes.set_aspect('auto')
    plt.xticks(fontsize=5)
    plt.xlabel(None)
    plt.ylabel(None)
    cbar = plt.colorbar(ax,shrink=0.5,label=xr_ar_f.name + ' ('+xr_ar_f.units+')', orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    if figformat == 'pdf': #needed to account for irregular behaviour with the alpha parameter when plotting a pdf file
       #fig.set_rasterized(True)
       print('Info: There is a problem with the alpha parameter when generating the figure on my local system. Correct this in future versions !')
    plt.savefig(savename_f,dpi=dpival_f)
    plt.close('all')
