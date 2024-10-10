import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool

class ClassificationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=0.8)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.8)
        self.fc1   = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2   = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # gcn + relu
        x = F.relu(self.conv1(x, edge_index))
        # pooling
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # readout
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        
        # sum for the pooling resluts
        x = x1 + x2
        
        # MLP for the classification 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        