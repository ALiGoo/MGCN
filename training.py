import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.mgcn import MGCN

log_path = "logs/mgcn"
batch_size = 48 * 24

# load data
loader = joblib.load("process_data/dataloader.pt")
graph = loader['graph']
station_info = pd.read_hdf('process_data/features_201307_201709.h5', key='info')

training_loader = DataLoader(loader['training_loader'], batch_size=batch_size, num_workers=16)
validation_loader = DataLoader(loader['validation_loader'], batch_size=batch_size, num_workers=16)
testing_loader = DataLoader(loader['testing_loader'], batch_size=batch_size, num_workers=16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_channels = 64
additinal_channels = 57
seq_len = 6
station_num = len(station_info)
graph = graph.to(device)
learning_rate = 1e-4

"""
建立model與optimizer與loss function
"""
model = MGCN(
    hidden_channels = hidden_channels,
    additinal_channels = additinal_channels,
    seq_len = seq_len,
    station_num = station_num,
).to(device)

min_rmse_epoch = 1
min_rmse = 100
# Training
for pre in [True, False]:
    criterion = nn.MSELoss()
    if pre:
        writer = SummaryWriter(f"{log_path}/pretrained")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        for param, name in zip(model.parameters(), model.state_dict().keys()):
            if "fc" not in name:
                param.requires_grad = False
        writer = SummaryWriter(f"{log_path}/finetune")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(1, 100 + 1, 1)):
        mean_loss = {
            "train_rmse": [],
            "train_rmsle": [],
            "train_len": 0,
            "val_rmse": [],
            "val_rmsle": [],
            "val_len": 0,
        }

        
        # training
        model.train()
        for inputs in tqdm(training_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            inputs.x = model(inputs, graph, pre_trained=pre)
            
            # backward
            loss = torch.sqrt(criterion(inputs.x, inputs.y))
            loss.backward()
            optimizer.step()

            # Store Loss
            mean_loss["train_rmse"].append(
                torch.sqrt(criterion(inputs.x[inputs.is_alive == 1], inputs.y[inputs.is_alive == 1])).item() * inputs.y.size(0)
            )
            mean_loss["train_rmsle"].append(
                torch.sqrt(criterion(torch.log1p(inputs.x[inputs.is_alive == 1]), torch.log1p(inputs.y[inputs.is_alive == 1]))).item()
                * inputs.y.size(0)
            )
            mean_loss["train_len"] += inputs.y.size(0)

        # validation
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(validation_loader):

                inputs = inputs.to(device)
                inputs.x = model(inputs, graph, pre_trained=pre)
                
                # Store Loss
                mean_loss["val_rmse"].append(
                    torch.sqrt(criterion(inputs.x[inputs.is_alive == 1], inputs.y[inputs.is_alive == 1])).item() * inputs.y.size(0)
                )
                mean_loss["val_rmsle"].append(
                    torch.sqrt(
                        criterion(torch.log1p(inputs.x[inputs.is_alive == 1]), torch.log1p(inputs.y[inputs.is_alive == 1]))
                    ).item()
                    * inputs.y.size(0)
                )
                mean_loss["val_len"] += inputs.y.size(0)

        train_rmse = np.sum(np.array(mean_loss["train_rmse"])) / mean_loss["train_len"]
        train_rmsle = np.sum(np.array(mean_loss["train_rmsle"])) / mean_loss["train_len"]
        val_rmse = np.sum(np.array(mean_loss["val_rmse"])) / mean_loss["val_len"]
        val_rmsle = np.sum(np.array(mean_loss["val_rmsle"])) / mean_loss["val_len"]
        if val_rmse < min_rmse:
            min_rmse_epoch = epoch
            min_rmse = val_rmse

        writer.add_scalar("RMSE/Train", train_rmse, epoch)
        writer.add_scalar("RMSE/Validation", val_rmse, epoch)
        writer.add_scalar("RMSLE/Train", train_rmsle, epoch)
        writer.add_scalar("RMSLE/Validation", val_rmsle, epoch)
        if pre:
            torch.save(model.state_dict(), f"{log_path}/pretrained/checkpoint_epoch{epoch}.pt")
        else:
            torch.save(model.state_dict(), f"{log_path}/finetune/checkpoint_epoch{epoch}.pt")


# Predict Testing Set 
model.load_state_dict(torch.load(f"logs/mgcn/finetune/checkpoint_epoch{min_rmse_epoch}.pt", map_location=device))
predicts = []
model.eval()
with torch.no_grad():
    for inputs in tqdm(testing_loader):
        inputs = inputs.to(device)
        inputs.x = model(inputs, graph, pre_trained=False)
        inputs.x = inputs.x[inputs.is_alive == 1]
        predicts.append(inputs.x.to('cpu').numpy())
predicts = np.concatenate(predicts, axis=0)
np.save("logs/mgcn/predicts.npy", predicts)