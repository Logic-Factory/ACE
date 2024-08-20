import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import torch
from torch_geometric.data import Data

from src.circuit.tag import Tag
from src.circuit.node import Node, Physics
    
class Circuit(object):
    """
    """
    def __init__(self, type_ckt:str):
        if type_ckt not in Tag.tags_circuit():
            raise ValueError("Invalid circuit type")
        self._type = type_ckt
        self._const0 = None # constant 0
        self._pis : list[Node]  = []      # primary input nodes: inputs
        self._pos : list[Node]  = []      # primary output nodes
        self._gates : list[Node]  = []    # internal logic nodes
        self._nodes : list[Node] = []     # all nodes: pis + gates + pos
        self._names : list[Node] = []     # store the name of each node
        self._node_map = {}               # [key, value], key is the index in old ckt, value is the index in self._nodes
        self._edges = []                  # store the edges of the circuit
        
        # attributes
        self._depths = []
        self._depth:int = 0

    def get_type(self):
        return self._type
    
    def node_at(self, idx:int) -> Node:
        return self._nodes[idx]
    
    def get_node(self, old_id:str) -> Node:
        if old_id not in self._node_map:
            raise ValueError("Invalid node old_id")
        idx = self._node_map[old_id]
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

    def add_fanin(self, idx:int, fanin:int):
        self._nodes[idx].add_fanin(fanin)
        self._edges.append((fanin, idx))

    def add_const0(self, old_id:str, old_name:str):
        idx = 0
        node = Node.make_const0(old_id, idx)
        self._const0 = node
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_pi(self, old_id:str, old_name:str):
        idx = len(self._nodes)
        node = Node.make_pi(old_id, idx)
        self._pis.append(old_id)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_po(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_po(old_id, idx, fanins)
        self._pos.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx
    
    def add_gate(self, type_node:str, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_node(type_node, old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_inverter(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_inverter(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_and2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_and2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_nand2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_nand2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_or2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_or2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_nor2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_nor2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_xnor2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xnor2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_xor2(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xor2(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_maj3(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_maj3(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_xor3(self, old_id:str, old_name:str, fanins:list = []):
        idx = len(self._nodes)
        node = Node.make_xor3(old_id, idx, fanins)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
        return idx

    def add_cell(self, old_id:str, old_name:str, fanins:list = [], physics:Physics=None):
        idx = len(self._nodes)
        node = Node.make_cell(old_id, idx, fanins, physics)
        self._gates.append(node)
        self._nodes.append(node)
        self._names.append(old_name)
        self._node_map[old_id] = idx
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

    def cal_depths(self):
        """ calculate the depth of the circuit
        """
        self._depths = [0] * len(self._nodes)
        
        for idx in range(len(self._nodes)):
            max_depth = 0
            def update_depth(node:Node):
                nonlocal max_depth
                max_depth = max(max_depth, self._depths[node.get_idx()] + 1)
            self.foreach_fanin(self._nodes[idx], update_depth)
            self._depths[idx] = max_depth

    def num_pis(self):
        return len(self._pis)

    def num_pos(self):
        return len(self._pos)
    
    def num_gates(self):
        return len(self._gates)
    
    def num_nodes(self):
        return len(self._nodes)
    
    def num_edges(self):
        return sum([len(node.get_fanins()) for node in self._nodes])

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

    def to_graphviz(self, file):
        pass

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
    idx_f = aig.add_po("f", [idx_g3])
    aig.add_fanin(idx_f, idx_g3)
    
    torch_data = aig.to_torch_geometric()
    print(torch_data)