import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class GCNConv(nn.Module):
    def __init__(self, station_num):
        super(GCNConv, self).__init__()
        self.weight_fusion = nn.Parameter(torch.Tensor(3, station_num, station_num))
        self.weight_gcn = nn.Parameter(torch.Tensor(station_num, station_num))
    
        self.reset_parameters()
    def reset_parameters(self):
        glorot(self.weight_fusion)
        glorot(self.weight_gcn)
    
    def forward(self, x, graph, ):
        fusion_graph = (graph * F.softmax(self.weight_fusion, dim=0)).sum(dim=0)
        x = fusion_graph @ self.weight_gcn @ x
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, hidden_channels, seq_len, station_num, in_channels=2, out_channels=2):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.station_num = station_num
        self.out_channels = out_channels

        self.conv = GCNConv(station_num)
        self.rnn = nn.LSTM(in_channels, hidden_channels)

    def forward(self, x, graph):
        x = F.relu(self.conv(x, graph), inplace=True)

        x = torch.cat([torch.zeros(self.seq_len-1,self.station_num,2, device=x.device), x])
        x = torch.cat([x[i - self.seq_len : i] for i in range(self.seq_len, x.size(0) + 1)], dim=1)
        _, (h, c) = self.rnn(x)
        return x, (h, c)

## Decoder

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        x, _ = self.rnn(x, (h, c))
        x = F.relu(self.lin(x[-1, :, :]), inplace=True)
        return x

## MGCN

class MGCN(nn.Module):
    def __init__(self, hidden_channels, additinal_channels, seq_len, station_num, in_channels=2, out_channels=2):
        super(MGCN, self).__init__()
        
        self.seq_len = seq_len
        self.encoder = Encoder(hidden_channels, seq_len, station_num,)
        self.decoder = Decoder(in_channels, hidden_channels, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels+additinal_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, inputs, graph, pre_trained=True):
        x, (h, c) = self.encoder(inputs.x, graph)
        if pre_trained:
            x = self.decoder(x[self.seq_len // 2 :, :, :], h, c)
        else:
            x = torch.cat([h[0], inputs.x_fc], 1)
            x = self.fc(x)
        return x