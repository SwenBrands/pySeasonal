# pySeasonal

A Python package to verify seasonal prediction systems against reanalysis
data or observations. It also provides real-time seasonal forecasts,
expressed as probabilities for a given variable to occur in the lower,
centre or upper tercile of the hindcast period. For skill and tercile
calculations, the hindcasts can be optionally combined with the past
forecasts. The package currently works with Copernicus Data Store products.

Author: Swen Brands, brandssf@ifcan.unican.es or swen.brands@gmail.com

The scripts have to be run in the following order:

1. regrid_obs.py #brings monthly obs to model grid and cuts out target
region (Medcof)

2. aggregate_hindcast.py #aggregates daily model data to monthly-mean
data and generates a single netCDF file per variable with dimensions
time x lead x member x lat x lon from the <yyyy><mm> netCDF files
provided by Predictia. This script also bring the netCDF variable names
in the model files to ERA5 standard (e.g. tas -> t2m or pr -> tp)

3. get_skill.py calculated hindcast skill measures and plots maps.


Credits
-------
This ongoing research work is being funded by the Ministry for the Ecological Transition and the Demographic Challenge (MITECO) and the European Commission NextGenerationEU (Regulation EU 2020/2094), through CSIC's Interdisciplinary Thematic Platform Clima (PTI-Clima).

![alt text](https://pti-clima.csic.es/wp-content/uploads/2023/11/Web-Gob-Min-CSIC-COLOR-LOGO-PNG-RGB-300pppCLIMA.png)

