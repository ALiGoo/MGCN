import pandas as pd
import numpy as np
import os
import json
from joblib import Parallel, delayed
from glob import glob
from datetime import datetime
import pickle
pickle.HIGHEST_PROTOCOL = 4

with open("raw_data/station_information.json", "r") as jsonfile:
    station_info = json.load(jsonfile)['data']['stations']
station_info = pd.json_normalize(station_info)[['station_id', 'name', 'lat', 'lon', ]]
station_info = station_info.astype({"station_id": np.int16, "lat": np.float32, "lon": np.float32})
station_info.rename(columns={'station_id':'stationid'}, inplace=True)

def read_citibike(path):
    
    dtype = {
        "tripduration": np.int32,
        "startstationlatitude": np.float32,
        "startstationlongitude": np.float32,
        "endstationlatitude": np.float32,
        "endstationlongitude": np.float32,
        "bikeid": np.int32,
        "gender": np.int8,
    }

    names = [
        "tripduration",
        "starttime",
        "stoptime",
        "startstationid",
        "startstationname",
        "startstationlatitude",
        "startstationlongitude",
        "endstationid",
        "endstationname",
        "endstationlatitude",
        "endstationlongitude",
        "bikeid",
        "usertype",
        "birthyear",
        "gender",
    ]

    df = pd.read_csv(
        path,
        header=0,
        names=names,
        dtype=dtype,
        engine="c",
        parse_dates=["starttime", "stoptime"],
    )
    df.dropna(subset=["startstationid", "endstationid"], inplace=True)
    df = df.astype({"startstationid": np.int16, "endstationid": np.int16})
    df.drop_duplicates(
        subset=["bikeid", "startstationid", "endstationid", "starttime", "stoptime",],
        inplace=True,
    )
    df.query(
        "startstationlatitude != 0 & endstationlatitude != 0 & starttime < stoptime",
        inplace=True,
    )
    df.drop(columns=['bikeid', 'birthyear'], inplace=True)

    return df

df_whole = Parallel(n_jobs=-1, verbose=50, backend="loky")(delayed(read_citibike)(path) for path in glob("raw_data/*trip*.csv"))
df_whole = pd.concat(df_whole)

# drop the stations that only return but not rent as the abnormal station in all records
startstationid = df_whole.startstationid.unique()
endstationid = df_whole.endstationid.unique()

print ("---------before---------")
print (f"Total startstationid: {len(startstationid)}")
print (f"Total endstationid: {len(endstationid)}")
    
pop_id = endstationid[~np.isin(endstationid, startstationid)]
df_whole.query(
    "startstationid not in @pop_id & endstationid not in @pop_id", inplace=True
)

startstationid = df_whole.startstationid.unique()
endstationid = df_whole.endstationid.unique()

print ("---------after---------")
print (f"Total startstationid: {len(startstationid)}")
print (f"Total endstationid: {len(endstationid)}")

lost_station = pd.concat([df_whole[['startstationid', 'startstationname', 'startstationlatitude', 'startstationlongitude']].drop_duplicates(subset='startstationid', keep="last").rename(columns={'startstationid':'stationid', 'startstationname':'name', 'startstationlatitude':'lat','startstationlongitude':'lon'}), df_whole[['endstationid', 'endstationname', 'endstationlatitude', 'endstationlongitude']].drop_duplicates(subset='endstationid', keep="last").rename(columns={'endstationid':'stationid', 'endstationname':'name', 'endstationlatitude':'lat','endstationlongitude':'lon'})])
lost_station = (
    lost_station
    .drop_duplicates(subset='stationid', keep="last")
    .query('stationid not in @station_info.stationid')
)

station_info = (
    pd.concat([station_info, lost_station])
    .query('stationid in @startstationid | stationid in @endstationid')
    .sort_values("stationid")
    .reset_index(drop=True)
)

# generate station info
starttime = df_whole.groupby("startstationid")["starttime"].min().dt.floor("D").values
stoptime = df_whole.groupby("endstationid")["stoptime"].min().dt.floor("D").values
station_info["earliest"] = np.where(starttime < stoptime, starttime, stoptime)

starttime = df_whole.groupby("startstationid")["starttime"].max().dt.floor("D").values
stoptime = df_whole.groupby("endstationid")["stoptime"].max().dt.floor("D").values
station_info["latest"] = np.where(starttime > stoptime, starttime, stoptime)

# Drop存活不到一天的場站
station_info.query("(latest - earliest).dt.days > 1", inplace=True)

# Drop存活不到一天的場站
df_whole.query(
    "startstationid in @station_info.stationid & endstationid in @station_info.stationid",
    inplace=True,
)
df_whole = df_whole[['starttime','stoptime', 'startstationid', 'endstationid', 'usertype', 'gender']]

# Weather

usecols = [
    "DATE",
    "HourlyDryBulbTemperature",
    "HourlyPrecipitation",
    "HourlyRelativeHumidity",
    "HourlyWindSpeed",
]
weather = pd.read_csv("raw_data/weather.csv", parse_dates=["DATE"], usecols=usecols)

# 選擇要的column並對na補值
weather = (
    weather.assign(
        DATE=weather.DATE.dt.ceil("H"),
        HourlyDryBulbTemperature=pd.to_numeric(
            weather.HourlyDryBulbTemperature, errors="coerce", downcast="float"
        ),
        HourlyPrecipitation=pd.to_numeric(
            weather.HourlyPrecipitation, errors="coerce", downcast="float"
        ),
        HourlyRelativeHumidity=pd.to_numeric(
            weather.HourlyRelativeHumidity, errors="coerce", downcast="float"
        ),
    )
    .groupby("DATE", as_index=False)
    .mean()
    .fillna(method="ffill")
    .rename(columns={"DATE": "time"})
)

df_whole.to_hdf("process_data/citibike_raw.h5", key="raw", mode="w")
station_info.to_hdf('process_data/citibike_raw.h5', key="info", mode="r+")
weather.to_hdf('process_data/citibike_raw.h5', key="weather", mode="r+")