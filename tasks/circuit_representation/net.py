import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class CircuitRankNet(nn.Module):
    def __init__(self, num_features):
        super(CircuitRankNet, self).__init__()
        self.activate = nn.Sigmoid()
        self.conv1 = SAGEConv(num_features, 2*num_features)
        self.conv2 = SAGEConv(2*num_features, 2*num_features)
        self.dropout = nn.Dropout(p=0.2)
        self.compare = nn.Sequential(
            nn.Linear(4*num_features, 2*num_features),  # Combine and compress features
            self.activate,
            nn.Linear(2*num_features, 1)  # Final decision
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
