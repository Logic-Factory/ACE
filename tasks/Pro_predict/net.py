from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

from torch.nn import LSTM, GRU
from typing import Optional

def subgraph(target_idx, edge_index,  dim=0):

    le_idx = []
    # print('edge_index.shape',edge_index.shape)
    for n in target_idx:
        ne_idx = edge_index[dim] == n
        le_idx += [ne_idx.nonzero().squeeze(-1)]
    le_idx = torch.cat(le_idx, dim=-1)
    lp_edge_index = edge_index[:, le_idx]
    return lp_edge_index


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

class TFMLP(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node’s neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, ouput_channels=64, edge_attr=None, mlp_post=None):
        super(TFMLP, self).__init__()
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1)

        self.msg_q = nn.Linear(in_channels, ouput_channels)
        self.msg_k = nn.Linear(in_channels, ouput_channels)
        self.msg_v = nn.Linear(in_channels, ouput_channels)


    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # print('edge_attr',edge_attr)
        return self.propagate(edge_index, x=x,edge_attr=None)

    def message(self, x_i, x_j, edge_attr, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key 
        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        # x_j -> value 
        t = self.msg_v(x_j) * a_j
        return t
    
    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out



class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5, sigmoid=False, tanh=False):
        super(MLP, self).__init__()
        '''
        The basic structure is refered from 
        '''
        assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
        self.norm_layer = nn.BatchNorm1d
        self.act_layer = nn.ReLU
        if p_drop > 0:
            self.dropout = nn.Dropout
        
        fc = []
        # 1st layer
        # print('dim_in',dim_in)
        # print('dim_hidden',dim_hidden)
        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))
        # last layer
        fc.append(nn.Linear(dim_hidden, dim_pred))
        # sigmoid
        if sigmoid:
            fc.append(nn.Sigmoid())
        if tanh:
            fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)
        self.init()

    def init(self):
        for m in  self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        out = self.fc(x)
        return out



class Gate_net(nn.Module):
  def __init__(self, args):
    super(Gate_net,self).__init__()
    self.args = args
    self.num_rounds = args.num_rounds
    self.device = args.device


    self.dim_hidden = args.dim_hidden
    self.dim_mlp = args.dim_mlp
    self.dim_pred = args.dim_pred
    
    self.aggr_and_strc =  TFMLP(in_channels=self.dim_hidden*1, ouput_channels=self.dim_hidden)
    self.aggr_and_func =  TFMLP(in_channels=self.dim_hidden*2, ouput_channels=self.dim_hidden)
    self.aggr_not_strc =  TFMLP(in_channels=self.dim_hidden*1, ouput_channels=self.dim_hidden)
    self.aggr_not_func =  TFMLP(in_channels=self.dim_hidden*1, ouput_channels=self.dim_hidden)

    self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
    self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
    self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
    self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)
    self.readout_prob = MLP(self.dim_hidden, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
    self.one = torch.ones(1).to(self.device)
    self.hf_emd_int = nn.Linear(1, self.dim_hidden)

  def forward(self,G):
    num_nodes = G.num_nodes
    num_layers_f = max(G.forward_level).item() + 1
    num_layers_b = max(G.backward_level).item() + 1
    #disable_encode
    hs_init = torch.zeros(num_nodes, self.dim_hidden)
    hs_init = hs_init.to(self.device)
    hf_init = self.hf_emd_int(self.one).view(1, -1)
    hf_init = hf_init.repeat(num_nodes, 1)
    preds = self._gru_forward(G, hs_init, hf_init, num_layers_f, num_layers_b)
    return preds
    
  def _gru_forward(self,G,hs_init,hf_init,num_layers_f, num_layers_b):
    # print('mlpgnn')
    G = G.to(self.device)
    edge_index = G.edge_index

    hs = hs_init.to(self.device)
    hf = hf_init.to(self.device)
    node_state = torch.cat([hs, hf], dim=-1)
    and_mask = G.gate == 6
    not_mask = G.gate == 4

    for _ in range(self.num_rounds):
        for level in range(1, num_layers_f):
            # forward layer
            layer_mask = G.forward_level == level

            # AND Gate
            l_and_node = G.forward_index[layer_mask & and_mask]
            if l_and_node.size(0) > 0:
                and_edge_attr = None
                and_edge_index = subgraph(l_and_node, edge_index, dim=1)
                # Update structure hidden state
                msg = self.aggr_and_strc(hs, and_edge_index,and_edge_attr)
                and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                hs[l_and_node, :] = hs_and.squeeze(0)
                # Update function hidden state
                msg = self.aggr_and_func(node_state, and_edge_index,and_edge_attr)
                and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                hf[l_and_node, :] = hf_and.squeeze(0)

            # NOT Gate
            l_not_node = G.forward_index[layer_mask & not_mask]
            if l_not_node.size(0) > 0:
                not_edge_attr = None
                not_edge_index  = subgraph(l_not_node, edge_index, dim=1)
                # Update structure hidden state
                msg = self.aggr_not_strc(hs, not_edge_index,not_edge_attr)
                not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                hs[l_not_node, :] = hs_not.squeeze(0)
                # Update function hidden state
                msg = self.aggr_not_func(hf, not_edge_index,not_edge_attr)
                not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                hf[l_not_node, :] = hf_not.squeeze(0)

            # Update node state
            node_state = torch.cat([hs, hf], dim=-1)

    node_embedding = node_state.squeeze(0)
    hs = node_embedding[:, :self.dim_hidden]
    hf = node_embedding[:, self.dim_hidden:]

    # Readout
    prob = self.readout_prob(hf)
    return prob

def get_model(args):
    return Gate_net(args)


    
if __name__=="__main__":
  model = get_model()
    
    
    
    