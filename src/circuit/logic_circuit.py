import torch
from torch_geometric.data import Data

from tag import Tag
from logic_node import LogicNode

class LogicGraph(object):
    def __init__(self, type_ckt:str):
        if type_ckt not in Tag.tags_circuit():
            raise ValueError("Invalid circuit type")
        self._type = type_ckt
        self._const0 = None # constant 0
        self._pis = []      # primary input nodes: const-0 + inputs
        self._pos = []      # primary output nodes
        self._gates = []    # internal logic nodes
        self._nodes = []    # all nodes: pis + gates + pos
        self._names = {}    # store the dict of the name its mapped node
        self._edges = []    

    def get_type(self):
        return self._type
    
    def is_gtech(self):
        return self._type == Tag.str_ckt_gtech()
    
    def is_aig(self):
        return self._type == Tag.str_ckt_aig()

    def is_xag(self):
        return self._type == Tag.str_ckt_xag()
    
    def is_xmg(self):
        return self._type == Tag.str_ckt_xmg()
    
    def is_mig(self):
        return self._type == Tag.str_ckt_mig()
    
    def is_xmg(self):
        return self._type == Tag.str_ckt_xmg()

    def add_edge(self, node_src:int, node_dest:int):
        self._edges.append((node_src, node_dest))

    def add_const0(self, name:str):
        idx = 0
        node = LogicNode.make_const0(name, idx)
        self._const0 = node
        self._nodes.append(node)
        return idx

    def add_pi(self, name:str):
        idx = len(self._nodes)
        node = LogicNode.make_pi(name, idx)
        self._pis.append(name)
        self._nodes.append(node)
        return idx

    def add_po(self, name:str, fanin0):
        idx = len(self._nodes)
        node = LogicNode.make_po(name, idx, fanin0)
        self._pos.append(node)
        self._nodes.append(node)
        self.add_edge(fanin0, idx)  # add fanin0 -> idx 
        return idx
    
    def add_gate(self, type_node:str, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = LogicNode.make_node(type_node, name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        
        # add edge between nodes
        for fanin in fanins:
            self.add_edge(fanin, idx)  # add fanin -> idx
        return idx

    def init_node_feature(self, type_node:str, size:int = 0):
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid node type")
        domains = LogicNode.node_domains()
        node_feature = []
        if size == 0 or size >= len(Tag.tags_node()):
            node_feature = [0] * len(domains)
            node_feature[domains[type_node]] = 1
        else:
            node_feature = [0] * size
            node_feature[ domains[type_node] % size ] = 1
        return node_feature

    def get_node_features(self, size:int = 0):
        """ Gen the node features for the node list
        """
        node_features = []
        for node in self._nodes:
            node_feature = self.init_node_feature( node.get_type(), size )
            node_features.append(node_feature)
        return torch.tensor(node_features, dtype=torch.float)

    def get_edge_index(self):
        """ Convert edge list to a tensor suitable for torch_geometric
        """
        edge_index = [[], []]
        for idx_src, idx_dest in self._edges:
            edge_index[0].append(idx_src)
            edge_index[1].append(idx_dest)
        return torch.tensor(edge_index, dtype=torch.long)

    def to_torch_geometric(self):
        return Data(x=self.get_node_features(), edge_index=self.get_edge_index())

if __name__ == '__main__':
    aig = LogicGraph(Tag.str_ckt_aig())
    # constant
    idx_const0 = aig.add_const0('const0')
    # primary inputs
    idx_a = aig.add_pi('a')
    idx_b = aig.add_pi('b')
    idx_c = aig.add_pi('c')
    idx_d = aig.add_pi('d')
    # internal gates
    idx_g1 = aig.add_gate(Tag.str_node_and2(), 'g1', [idx_a, idx_b])
    idx_g2 = aig.add_gate(Tag.str_node_and2(), 'g2', [idx_c, idx_d])
    idx_g3 = aig.add_gate(Tag.str_node_and2(), 'g3', [idx_g1, idx_g2])
    # primary outputs
    aig.add_po("f", idx_g3)
    
    torch_data = aig.to_torch_geometric()
    print(torch_data)