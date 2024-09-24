from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F




class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, feature, hidden):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(hidden, hidden)
        self.linear = nn.Linear(hidden,1)

    def forward(self, data):
        x, edge_index = data.x_feature, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        x = self.linear(x)
        return x
