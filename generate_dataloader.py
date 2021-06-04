import joblib
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from geopy.distance import geodesic

import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

dataset_period = [datetime.strptime('2013-07-01', '%Y-%m-%d'), datetime.strptime('2017-10-01', '%Y-%m-%d')]
test_period = [datetime.strptime('2017-10-01', '%Y-%m-%d') - timedelta(days=80), datetime.strptime('2017-10-01', '%Y-%m-%d')]
valid_period = [test_period[0] - timedelta(days=40), test_period[0]]
train_period = [dataset_period[0], valid_period[0]]
predict_time = "H"

features = pd.read_hdf('process_data/features_201307_201709.h5', key='features')
station_info = pd.read_hdf('process_data/features_201307_201709.h5', key="info")
alive_df = pd.read_hdf('process_data/features_201307_201709.h5', key='alive')
df_raw = pd.read_hdf('process_data/features_201307_201709.h5', key="raw")
df_raw['tripduration'] = (df_raw.stoptime - df_raw.starttime).dt.seconds
df_raw.query('starttime >= @train_period[0] & stoptime < @train_period[1]', inplace=True)

# Distance Graph

distance_graph = (
    station_info[["stationid", "lat", "lon"]]
    .assign(merge_key=1)
)
distance_graph = distance_graph.merge(distance_graph, on="merge_key").drop(
    "merge_key", axis=1
)
distance_graph["distance"] = distance_graph.apply(
    lambda x: geodesic((x.lat_x, x.lon_x), (x.lat_y, x.lon_y)).meters, axis=1
)
distance_graph = distance_graph.pivot(
    index="stationid_x", columns="stationid_y", values="distance"
)
distance_graph = distance_graph ** -1

for i in range(len(distance_graph)):
    distance_graph.iloc[i,i] = 0

distance_graph = distance_graph.replace([np.inf, -np.inf], np.nan)
distance_graph = distance_graph.fillna(0)

# Interaction Graph

interaction_graph = (
    df_raw[["startstationid", "endstationid"]]
    .groupby(["startstationid", "endstationid"])
    .size()
    .reset_index(name="counts")
)
interaction_graph = interaction_graph.pivot_table(
    index="startstationid", columns="endstationid", values="counts", fill_value=0
)
for i in range(len(interaction_graph)):
    interaction_graph.iloc[i,i] = 0

# Correlation Graph

correlation_graph = df_raw[["endstationid", "startstationid", "stoptime", "starttime"]]
correlation_graph = correlation_graph.assign(
    stoptime=correlation_graph.stoptime.dt.floor(predict_time),
    starttime=correlation_graph.starttime.dt.floor(predict_time),
)

correlation_graph_in = (
    correlation_graph.groupby(["endstationid", "stoptime"])
    .size()
    .reset_index(name="flow_in")
)
correlation_graph_in = correlation_graph_in.pivot_table(
    index="stoptime", columns="endstationid", values="flow_in", fill_value=0
)
correlation_graph_in = correlation_graph_in.corr("pearson")
correlation_graph_in[correlation_graph_in < 0] = 0

for i in range(len(correlation_graph_in)):
    correlation_graph_in.iloc[i,i] = 0

# Edge Index

def graph_norm(graph):
    deg = np.array(np.sum(graph, axis=1))
    deg = np.matrix(np.diag(deg)).astype(np.float32)
    deg_inv = np.power(deg,-1)
    deg_inv = np.where(np.isinf(deg_inv), 0, deg_inv)
    A_norm = np.matmul(deg_inv, graph) + np.identity(graph.shape[0])
    A_norm = torch.tensor(A_norm, dtype=torch.float32)
    
    return A_norm

distance_graph_norm = graph_norm(distance_graph.values)
correlation_graph_in_norm = graph_norm(correlation_graph_in.values)
interaction_graph_norm = graph_norm(interaction_graph.values)
graph = torch.stack([distance_graph_norm, interaction_graph_norm, correlation_graph_in_norm], dim=0)

# Split Training Validation Testing Set

station_num = len(station_info)
train_len = ((features.time >= train_period[0]) & (features.time < train_period[1])).sum() // station_num
val_len = ((features.time >= valid_period[0]) & (features.time < valid_period[1])).sum() // station_num
test_len = ((features.time >= test_period[0]) & (features.time < test_period[1])).sum() // station_num
print(f"train length:{train_len}\nvalidation length:{val_len}\ntest length:{test_len}")

columns = features.columns
columns.to_list()

gcn_column = ['flow_in_b1hour','flow_out_b1hour']
print(len(gcn_column))
gcn_column

fc_column = columns[~columns.str.contains('|'.join(['bike', 'flow', 'stationid', 'time', "y_in", "y_out"]))]
print(len(fc_column))
fc_column

y_column = ["y_in", "y_out"]
y_column

training_loader = []
for i in tqdm(np.arange(1, train_len + 1, 1)):
    x = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][gcn_column].values,
        dtype=torch.float,
    ).unsqueeze(0)
    y = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][y_column].values,
        dtype=torch.float,
    )
    x_fc = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][fc_column].values,
        dtype=torch.float,
    )
    is_alive = torch.tensor(
        alive_df.iloc[(i - 1) * station_num : i * station_num].is_alive.values,
        dtype=torch.int8,
    )
    training_loader.append(
        Data(
            x=x,
            y=y,
            x_fc=x_fc,
            is_alive=is_alive,
        )
    )

validation_loader = []
for i in tqdm(np.arange(train_len + 1, train_len + val_len + 1, 1)):
    x = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][gcn_column].values,
        dtype=torch.float,
    ).unsqueeze(0)
    y = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][y_column].values,
        dtype=torch.float,
    )
    x_fc = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][fc_column].values,
        dtype=torch.float,
    )
    is_alive = torch.tensor(
        alive_df.iloc[(i - 1) * station_num : i * station_num].is_alive.values,
        dtype=torch.int8,
    )
    validation_loader.append(
        Data(
            x=x,
            y=y,
            x_fc=x_fc,
            is_alive=is_alive,
        )
    )

testing_loader = []
for i in tqdm(
    np.arange(train_len + val_len + 1, train_len + val_len + test_len + 1, 1)
):
    x = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][gcn_column].values,
        dtype=torch.float,
    ).unsqueeze(0)
    
    y = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][y_column].values,
        dtype=torch.float,
    )
    x_fc = torch.tensor(
        features.iloc[(i - 1) * station_num : i * station_num][fc_column].values,
        dtype=torch.float,
    )
    is_alive = torch.tensor(
        alive_df.iloc[(i - 1) * station_num : i * station_num].is_alive.values,
        dtype=torch.int8,
    )
    testing_loader.append(
        Data(
            x=x,
            y=y,
            x_fc=x_fc,
            is_alive=is_alive,
        )
    )

joblib.dump(
    {
        'training_loader':training_loader,
        'validation_loader':validation_loader,
        'testing_loader':testing_loader,
        'graph':graph,
    },
    'process_data/dataloader.pt'
)