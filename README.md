# pySeasonal

A Python package to verify seasonal prediction systems against reanalysis
data or observations. It also provides real-time seasonal forecasts,
expressed as probabilities for a given variable to occur in the lower,
centre or upper tercile of the hindcast period. For skill and tercile
calculations, the hindcasts can be optionally combined with the past
forecasts. The package currently works with Copernicus Data Store products.
All input parameters are set at the start of the scripts and their use is
commented therein.

Author: Swen Brands, brandssf@ifcan.unican.es or swen.brands@gmail.com

The scripts have to be run sequentially in the following order:

1. regrid_obs.py #brings monthly observations / reanalysis data to the 
model grid and cuts out the target region (Medcof)

2. aggregate_hindcast.py #aggregates daily model data to monthly-mean
data and generates a single netCDF file per variable with dimensions
time x lead x member x lat x lon from the <yyyy><mm> netCDF files
provided by Predictia. This script also brings the netCDF variable names
in the model files to ERA5 standard (e.g. tas -> t2m or pr -> tp) and
performs the associated unit transformations.

3. get_skill_season.py #calculates hindcast skill measures for the
selected spatial domain, period, season and lead time. Also calculates
the climatological tercile thresholds needed in pred2tercile.py

4. plot_seasonal_validation_results.py #plots the results obtained with
get_skill_season.py and generates a netCDF file containing binary skill
masks

5. pred2tercile.py #aggregates the daily GCM data of a given foreccast
to seasonal average predictions and transforms them into forecast
probabilities per tercile. These are then put into netCDF format

6. functions_seasonal.py #contains the functions used by the afore-
mentioned scripts


Secondary scripts:
- downloadme_era5_monthly_single_level.py #script to download monthly
ERA5 data form CDS with the CDS API; for variables on single levels

- downloadme_era5_monthly_pressure_level.py #script to download monthly
ERA5 data form CDS with the CDS API; for variables on pressure levels


Deprecated scripts:
- get_skill.py
- map_results.py


Credits
-------
This ongoing research work is being funded by the Ministry for Ecological Transition and Demographic Challenge (MITECO) and the European Commission NextGenerationEU (Regulation EU 2020/2094), through CSIC's Interdisciplinary Thematic Platform Clima (PTI-Clima).

![alt text](https://pti-clima.csic.es/wp-content/uploads/2023/11/Web-Gob-Min-CSIC-COLOR-LOGO-PNG-RGB-300pppCLIMA.png)

