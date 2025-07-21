import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGPooling
from torch.nn import BatchNorm1d

class CircuitRankNet(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(CircuitRankNet, self).__init__()
        self.activate = nn.Sigmoid()
        
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
    
    # def logic_embedding(self, logics):
    #     # Logic is a batch of strings ["aig", "xag", "mig", "gtg"], generate its one-hot embedding
    #     logic_pool = ["aig", "xag", "mig", "gtg"]
    #     # Create a tensor to hold the one-hot encoding for each logic in the batch
    #     one_hot = torch.zeros(len(logics), len(logic_pool), dtype=torch.float32)
    #     # Fill the one-hot tensor
    #     for i, logic in enumerate(logics):
    #         index = logic_pool.index(logic)
    #         one_hot[i, index] = 1.0
    #     return one_hot
    
    def forward(self, data0:Data, data1:Data):
        # Extract features for each graph
        f0 = self.graph_embedding(data0)
        f1 = self.graph_embedding(data1)
        conbination_features = torch.cat([f0, f1], dim=1)
        prob = self.activate( self.compare(conbination_features) )
        return prob.squeeze(-1)


Logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]

class CircuitRankNet2(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(CircuitRankNet2, self).__init__()
        self.activate1 = nn.ReLU()
        self.activate2 = nn.Sigmoid()
        
        self.conv1 = SAGEConv(dim_input, dim_hidden)
        self.conv2 = SAGEConv(dim_hidden, dim_hidden)
        self.bn1 = BatchNorm1d(dim_hidden)
        self.bn2 = BatchNorm1d(dim_hidden)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # Feature fusion layer (combines graph and logic features)
        self.fusion = nn.Linear(2*dim_hidden + len(Logics), 2 * dim_hidden)
        
        self.pool = SAGPooling(dim_hidden, ratio=0.5)
        
        self.compare = nn.Sequential(
            nn.Linear(8 * dim_hidden, dim_hidden),
            self.activate1,
            nn.Linear(dim_hidden, 1),
            self.activate2
        )
        
    def graph_embedding(self, data):
        f = self.conv1(data.x, data.edge_index)
        f = self.bn1(f)
        f = self.dropout(f)
        f = self.activate1(f)
        
        f = self.conv2(f, data.edge_index)
        f = self.bn2(f)
        f = self.dropout(f)
        f = self.activate1(f)

        # f, edge_index, _, batch, _, _ = self.pool(f, data.edge_index, batch=data.batch)

        f_mean = global_mean_pool(f, data.batch)
        f_max = global_max_pool(f, data.batch)        
        f_combined = torch.cat([f_mean, f_max], dim=1)
        return f_combined
    
   
    def fuse_feature(self, graph_embed:torch.Tensor, logic_embed: torch.Tensor) -> torch.Tensor:
        """
        Fuse graph embeddings with one-hot encoded logic features.

        Args:
            graph_embed (torch.Tensor): Graph embedding (shape: batch_size, 2 * dim_hidden).
            logic (torch.Tensor): One-hot encoded logic vector (shape: batch_size, num_logics).

        Returns:
            torch.Tensor: Fused features (shape: batch_size, dim_hidden).
        """   
        combined = torch.cat([graph_embed, logic_embed], dim=1)
        fused = self.fusion(combined)
        return fused
    
    def forward(self, data0: Data, logic0_embed: torch.Tensor, data1: Data, logic1_embed: torch.Tensor):
        f0 = self.graph_embedding(data0)
        f1 = self.graph_embedding(data1)
        
        # fusion graph and logic features
        fused0 = self.fuse_feature(f0, logic0_embed)
        fused1 = self.fuse_feature(f1, logic1_embed)
                
        # combine features for comparison
        combined_features = torch.cat([
            fused0,
            fused1,
            torch.abs(fused0 - fused1),
            fused0 * fused1
        ], dim=1)
        
        prob = self.compare(combined_features)
        return prob.squeeze(-1)

    # def contrastive_loss(self, f0, f1, target):
    #     distance = torch.norm(f0 - f1, p=2, dim=1)
    #     loss = (1 - target) * (distance ** 2) + target * F.relu(self.margin - distance) ** 2
    #     return loss.mean()

    # def mutual_information_loss(f0, f1, temperature=0.1):
    #     # 归一化嵌入
    #     f0 = F.normalize(f0, p=2, dim=1)
    #     f1 = F.normalize(f1, p=2, dim=1)
    #     # 计算相似度矩阵
    #     sim_matrix = torch.mm(f0, f1.t()) / temperature
    #     labels = torch.arange(f0.size(0))
    #     loss = F.cross_entropy(sim_matrix, labels)
    #     return loss
