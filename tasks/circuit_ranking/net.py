import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool
from torch_geometric.nn import SAGPooling

class CircuitRankNet(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(CircuitRankNet, self).__init__()
        self.activate = nn.Sigmoid()
        
        # # 定义 MLP 用于 GINConv
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(dim_input, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, dim_hidden)
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, dim_hidden)
        # )        
        # self.conv1 = GINConv(self.mlp1)
        # self.conv2 = GINConv(self.mlp2)


        self.conv1 = GCNConv(dim_input, dim_hidden)
        self.conv2 = GCNConv(dim_hidden, dim_hidden)
        self.dropout = nn.Dropout(p=0.2)
        self.compare = nn.Sequential(
            nn.Linear(2*dim_hidden, dim_hidden),  # Combine and compress features
            self.activate,
            nn.Linear(dim_hidden, 1)              # Final decision
        )
        
    def graph_embedding(self, data):
        f = self.conv1(data.x, data.edge_index)
        f = self.dropout(f)
        f = self.conv2(f, data.edge_index)
        f = self.dropout(f)
        f = global_mean_pool(f, data.batch)
        return f
    
    def forward(self, data0:Data, data1:Data):
        # Extract features for each graph
        f0 = self.graph_embedding(data0)
        f1 = self.graph_embedding(data1)
        conbination_features = torch.cat([f0, f1], dim=1)
        prob = self.activate( self.compare(conbination_features) )
        return prob.squeeze(-1)
