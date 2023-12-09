#!/usr/bin/env python

import cdsapi
c = cdsapi.Client()

startyear = 1940
endyear = 2022
variable = 'mean_sea_level_pressure'  #'total_precipitation', '2m_temperature', 'sea_surface_temperature', mean_sea_level_pressure
#tardir = '/media/swen/ext_disk2/datos/OBSdata/era5/mon'
tardir = '/home/swen/datos/OBSData/era5/mon'

## execute ################################################################
if variable == '2m_temperature':
    varalias = 't2m'
elif variable == 'total_precipitation':
    varalias = 'tp'
elif variable == 'sea_surface_temperature':
    varalias = 'sst'
elif variable == '10m_wind_speed':
    varalias = 'si10'
elif variable == 'surface_solar_radiation_downwards':
    varalias = 'ssrd'
elif variable == 'mean_sea_level_pressure':
    varalias = 'msl'
else:
    raise Exception('ERROR: unknown entry for <variable>')
print('INFO: Downloading '+variable+' with alias '+varalias+'...')  

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'variable': variable,
        'year': [
            '1940', '1941', '1942',
            '1943', '1944', '1945',
            '1946', '1947', '1948',
            '1949', '1950', '1951',
            '1952', '1953', '1954',
            '1955', '1956', '1957',
            '1958', '1959', '1960',
            '1961', '1962', '1963',
            '1964', '1965', '1966',
            '1967', '1968', '1969',
            '1970', '1971', '1972',
            '1973', '1974', '1975',
            '1976', '1977', '1978',
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'product_type': 'monthly_averaged_reanalysis',
    },
    tardir+'/era5_mon_'+varalias+'_'+str(startyear)+'_'+str(endyear)+'.nc')
