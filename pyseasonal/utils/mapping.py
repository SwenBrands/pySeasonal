import numpy as np


subperiod_years = {
    "mod2strong_nino_oni": {
        "years": [1982, 1983, 1986, 1987, 1991, 1992, 1997, 1998, 2009, 2010, 2015, 2016],
        "msg": (
            'The model is verified for moderate and strong El Niño years based on '
            'ONI index only: {years}'
        )
    },
    "mod2strong_nina_oni": {
        "years": [
            1984, 1985, 1988, 1989, 1999, 2000, 2007, 2008, 2010, 2011, 2020, 2021, 2022
        ],
        "msg": (
            'The model is verified for moderate and strong La Niña years based on '
            'ONI index only: {years}'
        )
    },
    "enso_nino_noaa": {
        "years": [1983, 1987, 1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016],
        "msg": (
            'The model is verified for the El Niño years declared by NOAA at '
            'https://psl.noaa.gov/enso/past_events.html: {years}')
    },
    "enso_nina_noaa": {
        "years": [1989, 1999, 2000, 2008, 2011, 2012, 2021, 2022],
        "msg": (
            'The model is verified for the La Niña years declared by NOAA at '
            'https://psl.noaa.gov/enso/past_events.html : {years}'
        )
    },
    "enso_neutral_noaa": {
        "years": [
            1981, 1982, 1984, 1985, 1986, 1990, 1991, 1993, 1994, 1996, 1997, 2001,
            2002, 2004, 2005, 2006, 2009, 2013, 2014, 2015, 2017, 2018, 2019, 2020
        ],
        "msg": (
            'The model is verified for the neutral ENSO years declared by NOAA at '
            'https://psl.noaa.gov/enso/past_events.html : {years}'
        )
    },
    "qbo50_pos": {
        "years": [
            1981, 1983, 1985, 1986, 1988, 1991, 1993, 1995, 1997, 1999,
            2000, 2002, 2004, 2009, 2011, 2014, 2017, 2019, 2021, 2023
        ],
        "msg": (
            'The model is verified for positive QBO-50 years derived from '
            'https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: {years}'
        )
    },
    "qbo50_neg": {
        "years": [
            1982, 1984, 1987, 1989, 1992, 1994, 1996, 1998, 2001,
            2003, 2005, 2007, 2010, 2012, 2015, 2018, 2022
        ],
        "msg": (
            'The model is verified for negative QBO-50 years derivded from '
            'https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: {years}'
        )
    },
    "qbo50_trans": {
        "years": [1990, 2006, 2008, 2013, 2016, 2020],
        "msg": (
            'The model is verified for transition QBO-50 years derived from '
            'https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index only: {years}'
        )
    },
    "none": {
        "years": np.arange(1981, 2023, 1),
        "msg": (
            'The full overlapping period between observations and model data is '
            'used for verification.'
        )
    },
}

