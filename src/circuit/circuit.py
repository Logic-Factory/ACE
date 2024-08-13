import torch
from torch_geometric.data import Data

from .tag import Tag
from .node import Physics, Node  
    
class Circuit(object):
    """
    """
    def __init__(self, type_ckt:str):
        if type_ckt not in Tag.tags_circuit():
            raise ValueError("Invalid circuit type")
        self._type = type_ckt
        self._const0 = None # constant 0
        self._pis = []      # primary input nodes: inputs
        self._pos = []      # primary output nodes
        self._gates = []    # internal logic nodes
        self._nodes = []    # all nodes: pis + gates + pos
        self._names = {}    # store the dict of the name its mapped node
        self._edges = []    

    def get_type(self):
        return self._type
    
    def node_at(self, idx:int) -> Node:
        return self._nodes[idx]
    
    def get_node(self, name:str) -> Node:
        if name not in self._names:
            raise ValueError("Invalid node name")
        idx = self._names[name]
        return self._nodes[idx]
    
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
    
    def is_cell(self):
        return self._type == Tag.str_ckt_cell()

    def add_edge(self, node_src:int, node_dest:int):
        self._edges.append((node_src, node_dest))

    def add_fanin(self, idx:int, fanin:int):
        self._nodes[idx].add_fanin(fanin)

    def add_const0(self, name:str):
        idx = 0
        node = Node.make_const0(name, idx)
        self._const0 = node
        self._nodes.append(node)
        self._names[name] = idx
        return idx

    def add_pi(self, name:str):
        idx = len(self._nodes)
        node = Node.make_pi(name, idx)
        self._pis.append(name)
        self._nodes.append(node)
        self._names[name] = idx
        return idx

    def add_po(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_po(name, idx, fanins)
        self._pos.append(node)
        self._nodes.append(node)
        # self.add_edge(fanin0, idx)  # add fanin0 -> idx 
        self._names[name] = idx
        return idx
    
    def add_gate(self, type_node:str, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_node(type_node, name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        
        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_inverter(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_inverter(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_and2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_and2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        for fanin in fanins:
            self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_nand2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_nand2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_or2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_or2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_nor2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_nor2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_xnor2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xnor2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        for fanin in fanins:
            self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_xor2(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xor2(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_maj3(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_maj3(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_xor3(self, name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xor3(name, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def add_cell(self, name:str, fanins:list = [], physics:Physics=None):
        idx = len(self._nodes)
        node = Node.make_cell(name, idx, fanins, physics)
        self._gates.append(node)
        self._nodes.append(node)

        # for fanin in fanins:
        #     self.add_edge(fanin, idx)  # add fanin -> idx
        self._names[name] = idx
        return idx

    def foreach_pi(self, func):
        for node in self._pis:
            func(node)

    def forach_po(self, func):
        for node in self._pos:
            func(node)
            
    def foreach_gate(self, func):
        for node in self._gates:
            func(node)

    def foreach_node(self, func):
        for node in self._nodes:
            func(node)
    
    def foreach_fanin(self, node:Node, func):
        for fanin in node.get_fanins():
            func(self.node_at(fanin))

    def init_node_feature(self, type_node:str, size:int = 0):
        # TODO: use different node embedding method for different circuit type
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid node type")
        domains = Node.node_domains()
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
    aig = Circuit(Tag.str_ckt_aig())
    # constant
    idx_const0 = aig.add_const0('const0')
    # primary inputs
    idx_a = aig.add_pi('a')
    idx_b = aig.add_pi('b')
    idx_c = aig.add_pi('c')
    idx_d = aig.add_pi('d')
    # internal gates
    idx_g1 = aig.add_gate(Tag.str_node_and2(), 'g1', [idx_a, idx_b])
    aig.add_fanin(idx_g1, idx_a)
    aig.add_fanin(idx_g1, idx_b)
    idx_g2 = aig.add_gate(Tag.str_node_and2(), 'g2', [idx_c, idx_d])
    aig.add_fanin(idx_g2, idx_c)
    aig.add_fanin(idx_g2, idx_d)
    idx_g3 = aig.add_gate(Tag.str_node_and2(), 'g3', [idx_g1, idx_g2])
    aig.add_fanin(idx_g3, idx_g1)
    aig.add_fanin(idx_g3, idx_g2)
    # primary outputs
    idx_f = aig.add_po("f", idx_g3)
    aig.add_fanin(idx_f, idx_g3)
    
    torch_data = aig.to_torch_geometric()
    print(torch_data)