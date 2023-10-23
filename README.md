# pySeasonal
A Python package to verify seasonal prediction systems against observations

This package of Python scripts and functions evaluates the skill of the
Copernicus Data Store seasonal prediction products against reanalyses 
or observations. The scripts have to be run in the following order:

1. regrid_obs.py #brings monthly obs to model grid and cuts out target
region (Medcof)

2. aggregate_hindcast.py #aggregates daily model data to monthly-mean
data and generates a single netCDF file per variable with dimensions
time x lead x member x lat x lon from the <yyyy><mm> netCDF files
provided by Predictia. This script also bring the netCDF variable names
in the model files to ERA5 standard (e.g. tas -> t2m or pr -> tp)

3. get_skill.py calculated hindcast skill measures and plots maps.
