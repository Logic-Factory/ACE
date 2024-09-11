import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from .graphml2data import node_types, edge_types, GraphMLDataset

node_feratures = len(node_types)

class GCN(torch.nn.Module):
    def __init__(self, nodeEmbeddingDim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_feratures, nodeEmbeddingDim)
        self.conv2 = GCNConv(nodeEmbeddingDim, nodeEmbeddingDim)
        self.conv3 = GCNConv(nodeEmbeddingDim, nodeEmbeddingDim)
        self.lin = torch.nn.Linear(nodeEmbeddingDim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)    # 使用全局平均池化获得图的嵌入
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x