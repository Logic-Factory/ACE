import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, MessagePassing, GCNConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_max_pool, global_mean_pool

full_node_feature_dims = 21
full_synthesis_feature_dims = 14

import torch
import torch.nn as nn

class NodeEncoder(nn.Module):
    """
    Node encoder class to encode node features into embedding vectors.
    """
    
    def __init__(self, emb_dim):
        """
        Initialize the node encoder.

        Parameters:
        emb_dim (int): Dimension of the embedding vector.
        """
        super(NodeEncoder, self).__init__()
        
        # Define the node type embedding layer
        self.node_type_embedding = nn.Embedding(full_node_feature_dims, emb_dim)
        
        # Initialize the weights of the embedding layer using Xavier uniform distribution
        nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # Use the embedding layer to convert node features into embedding vectors
        x_embedding = self.node_type_embedding(x)
        
        return x_embedding

class GNN(torch.nn.Module):
    """
    Graph Neural Network (GNN) class to encode graph structure and node features.

    """
    def __init__(self, node_encoder, input_dim, emb_dim=64, gnn = 'gin'):
        super(GNN, self).__init__()
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder
        if gnn == 'gcn':
            self.conv1 = GCNConv(full_node_feature_dims, emb_dim)
            self.conv2 = GCNConv(emb_dim, emb_dim)
        elif gnn == 'sage':
            self.conv1 = SAGEConv(full_node_feature_dims, emb_dim)
            self.conv2 = SAGEConv(emb_dim, emb_dim)
        elif gnn == 'gin':
            self.conv1 = GINConv(
                Sequential(Linear(full_node_feature_dims, emb_dim), BatchNorm1d(emb_dim), ReLU(),
                        Linear(emb_dim, emb_dim), ReLU()))
            self.conv2 = GINConv(
                Sequential(Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
                        Linear(emb_dim, emb_dim), ReLU()))

        self.batch_norm1 = torch.nn.BatchNorm1d(emb_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(emb_dim)

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        h = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        h = self.batch_norm2(self.conv2(h, edge_index))

        xF = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=1)
        return xF

class SynthFlowEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(SynthFlowEncoder, self).__init__()
        self.synth_emb = torch.nn.Embedding(full_synthesis_feature_dims, emb_dim)
        torch.nn.init.xavier_uniform_(self.synth_emb.weight.data)

    def forward(self, x):
        x_embedding = self.synth_emb(x[:, 0])
        for i in range(1, x.shape[1]):
            x_embedding = torch.cat((x_embedding, self.synth_emb(x[:, i])), dim=1)
        return x_embedding

class SynthConv(torch.nn.Module):
    def __init__(self, inp_channel=1,out_channel=3,ksize=6,stride_len=1):
        super(SynthConv, self).__init__()
        self.conv1d = torch.nn.Conv1d(inp_channel,out_channel,kernel_size=(ksize,),stride=(stride_len,))

    def forward(self, x):
        x = x.reshape(-1,1,x.size(1)) # Convert [4,60] to [4,1,60]
        x = self.conv1d(x)
        return x.reshape(x.size(0),-1) # Convert [4,3,55] to [4,165]

class SynthNet(torch.nn.Module):
    def __init__(self, node_encoder, synth_encoder, n_classes, synth_input_dim, node_input_dim, gnn_embed_dim=128, hidden_dim=256):
        super(SynthNet, self).__init__()
        self.node_encoder = node_encoder
        self.synth_encoder = synth_encoder
        self.hidden_dim = hidden_dim
        self.synth_enc_outdim = 32
        self.gnn_emb_dim = gnn_embed_dim
        self.num_layers = 4
        
        self.synconv_in_channel = 1
        self.synconv_out_channel = 1
        self.synconv_stride_len = 3

        # Synth Conv1 output
        self.synconv1_ks = 12
        self.synconv1_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv1_ks)/self.synconv_stride_len

        # Synth Conv2 output
        self.synconv2_ks = 15
        self.synconv2_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv2_ks) / self.synconv_stride_len

        # Synth Conv3 output
        self.synconv3_ks = 18
        self.synconv3_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv3_ks) / self.synconv_stride_len

        # Synth Conv4 output
        self.synconv4_ks = 21
        self.synconv4_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv4_ks) / self.synconv_stride_len

        # Multiplier by 2 since each gate and node type has same encoding out dimension
        # self.gnn = GNN(self.node_encoder,self.node_enc_outdim*2)
        # Node encoding has dimension 3 and number of incoming inverted edges has dimension 1
        self.synth_conv1 = SynthConv(self.synconv_in_channel,self.synconv_out_channel,ksize=self.synconv1_ks,stride_len=self.synconv_stride_len)
        self.synth_conv2 = SynthConv(self.synconv_in_channel,self.synconv_out_channel,ksize=self.synconv2_ks,stride_len=self.synconv_stride_len)
        self.synth_conv3 = SynthConv(self.synconv_in_channel,self.synconv_out_channel,ksize=self.synconv3_ks,stride_len=self.synconv_stride_len)
        self.synth_conv4 = SynthConv(self.synconv_in_channel,self.synconv_out_channel,ksize=self.synconv4_ks,stride_len=self.synconv_stride_len)

        self.fcs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # GNN + (synthesis flow encoding + synthesis convolution)
        self.in_dim_to_fcs = int(self.gnn_emb_dim + self.synconv1_out_dim_flatten + self.synconv3_out_dim_flatten + self.synconv2_out_dim_flatten + self.synconv4_out_dim_flatten)
        self.fcs.append(torch.nn.Linear(1090,self.hidden_dim))
        #self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        for layer in range(1, self.num_layers-1):
            self.fcs.append(torch.nn.Linear(self.hidden_dim,self.hidden_dim))
            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.fcs.append(torch.nn.Linear(self.hidden_dim, 1))
        self.gnn = GNN(self.node_encoder, node_input_dim, gnn_embed_dim)
        self.fc = torch.nn.Linear(gnn_embed_dim + synth_input_dim, hidden_dim)  # Adjusted input size
        self.output_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch_data):
        graphEmbed = self.gnn(batch_data)
        synthFlow = batch_data.seq
        synEmbed = self.synth_encoder(synthFlow.reshape(-1, 20))
        synconv1_out = self.synth_conv1(synEmbed)
        synconv2_out = self.synth_conv2(synEmbed)
        synconv3_out = self.synth_conv3(synEmbed)
        synconv4_out = self.synth_conv4(synEmbed)

        # Concatenate all inputs
        concatenatedInput = torch.cat([graphEmbed, synconv1_out, synconv2_out, synconv3_out,synconv4_out], dim=1)
        x = F.relu(self.fcs[0](concatenatedInput))

        # Pass inputs through the remaining linear layers
        for layer in range(1, self.num_layers-1):
            x = F.relu(self.fcs[layer](x))

        # Pass inputs through the last linear layer and return the output
        x = self.fcs[-1](x)
        return x