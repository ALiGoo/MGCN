import joblib
import warnings
import pandas as pd
import numpy as np
import gc
from datetime import datetime, timedelta
from glob import glob
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing

warnings.filterwarnings("ignore")

dataset_period = [datetime.strptime('2013-07-01', '%Y-%m-%d'), datetime.strptime('2017-10-01', '%Y-%m-%d')]
test_period = [datetime.strptime('2017-10-01', '%Y-%m-%d') - timedelta(days=80), datetime.strptime('2017-10-01', '%Y-%m-%d')]
valid_period = [test_period[0] - timedelta(days=40), test_period[0]]
train_period = [dataset_period[0], valid_period[0]]
predict_time = "H"

# Load Data

df_raw = pd.read_hdf("process_data/citibike_raw.h5", key="raw")
station_info = pd.read_hdf("process_data/citibike_raw.h5", key="info")
weather = pd.read_hdf("process_data/citibike_raw.h5", key="weather")

station_info.query('earliest < @train_period[1]', inplace=True)
df_raw.query('starttime >= @dataset_period[0] & stoptime < @dataset_period[1] & startstationid in @station_info.stationid & endstationid in @station_info.stationid', inplace=True)

# drop the stations that only return but not rent as the abnormal station in all records
startstationid = df_raw.startstationid.unique()
endstationid = df_raw.endstationid.unique()

print ("---------before---------")
print (f"Total startstationid: {len(startstationid)}")
print (f"Total endstationid: {len(endstationid)}")
print (f"Total info station: {len(station_info)}")
    
pop_id = endstationid[~np.isin(endstationid, startstationid)]
df_raw.query(
    "startstationid not in @pop_id & endstationid not in @pop_id", inplace=True
)
station_info.query('stationid in @df_raw.startstationid | stationid in @df_raw.endstationid', inplace=True)

startstationid = df_raw.startstationid.unique()
endstationid = df_raw.endstationid.unique()

print ("---------after---------")
print (f"Total startstationid: {len(startstationid)}")
print (f"Total endstationid: {len(endstationid)}")
print (f"Total info station: {len(station_info)}")

# Flow

flow_in = (
    df_raw.assign(stoptime=df_raw.stoptime.dt.floor(predict_time))
    .groupby(["endstationid", "stoptime"])
    .size()
    .reset_index(name="flow_in")
    .rename(columns={"stoptime": "time", "endstationid": "stationid"})
)

flow_out = (
    df_raw.assign(starttime=df_raw.starttime.dt.floor(predict_time))
    .groupby(["startstationid", "starttime"])
    .size()
    .reset_index(name="flow_out")
    .rename(columns={"starttime": "time", "startstationid": "stationid"})
)

features = pd.date_range(
    datetime.strptime("2013-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2017-09-30 23:00:00", "%Y-%m-%d %H:%M:%S"),
    freq=predict_time,
)
features = list(product(features, station_info.stationid))
features = pd.DataFrame(features, columns=["time", "stationid"])
features = features.merge(flow_in, on=["time", "stationid"], how="left")
features = features.merge(flow_out, on=["time", "stationid"], how="left")
features.fillna(0, inplace=True)
features[features.columns[1:]] = features[features.columns[1:]].astype("int16")

del flow_in, flow_out
gc.collect()

# Shift features

features = (
    features.assign(is_weekend=features.time.dt.dayofweek >= 5)
    .astype({"is_weekend": "int8"})
    .set_index(["time", "is_weekend", "stationid"])
)

features = pd.concat(
    [
        features.rename(columns={"flow_in": "y_in", "flow_out": "y_out"}),
        features[["flow_in", "flow_out"]].groupby(level=2).shift(1, fill_value=-1).add_suffix("_b1hour"),
    ],
    axis=1,
).iloc[len(station_info) :]

# Do Dummy

features = features.reset_index()

features = features.assign(
    month=features.time.dt.month,
    dayofweek=features.time.dt.dayofweek,
    hour=features.time.dt.hour,
)

features = pd.get_dummies(
    features, columns=["month", "dayofweek", "hour"], drop_first=True, dtype=np.int8
)

# Weather

# Z-Score normalize
norm_col = weather.columns[1:]
weather[norm_col] = (weather[norm_col] - weather[norm_col].mean()) / weather[norm_col].std()

time = pd.date_range(
    datetime.strptime("2013-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2017-09-30 23:00:00", "%Y-%m-%d %H:%M:%S"),
    freq=predict_time,
)
time = pd.DataFrame(time,columns=['time'])
weather = time.merge(weather, on=["time"], how="left")
weather = weather.fillna(method="ffill")

# Concat weather to features
features = features.merge(weather, on="time", how="left")

# Alive Datafeame

alive_df = features[["stationid", "time"]]

for stationid in tqdm(station_info.stationid):
    condition = (
        alive_df.loc[alive_df.stationid == stationid, "time"]
        > station_info.loc[station_info.stationid == stationid, "earliest"].values[0]
    ) & (
        alive_df.loc[alive_df.stationid == stationid, "time"]
        < station_info.loc[station_info.stationid == stationid, "latest"].values[0]
    )
    alive_df.loc[alive_df.stationid == stationid, "is_alive"] = np.where(
        condition, 1, 0
    )
alive_df["is_alive"] = alive_df["is_alive"].astype("int8")

features.to_hdf('process_data/features_201307_201709.h5', key="features", mode="w")
df_raw.to_hdf('process_data/features_201307_201709.h5', key="raw", mode="r+")
station_info.to_hdf('process_data/features_201307_201709.h5', key="info", mode="r+")
alive_df.to_hdf('process_data/features_201307_201709.h5', key="alive", mode="r+")