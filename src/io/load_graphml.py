import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import argparse
import networkx as nx

from src.circuit.tag import Tag
from src.circuit.node import Node
from src.circuit.circuit import Circuit


def load_graphml(filename:str, type_ckt:str) -> Circuit:
    """
    Load a graphml file and return a LogicGraph/CellGraph object
    :param filename: the name of the file to load
    :return: a LogicGraph object
    """
    if type_ckt not in Tag.tags_circuit():
        raise ValueError("Invalid circuit type")

    raw_graph = nx.read_graphml(filename)
    circuit = Circuit(type_ckt)
    
    # only add the nodes, leave the fanins alone
    for id, attr in raw_graph.nodes(data=True):
        node_type = attr.get('type')
        assert node_type in Tag.tags_node(), f"Invalid node type: {node_type}"

        node_id = str(id)
        node_name = attr.get('name')
        if node_type == Tag.str_node_const0():
            circuit.add_const0(node_id, node_name)
        elif node_type == Tag.str_node_pi():
            circuit.add_pi(node_id, node_name)
        elif node_type == Tag.str_node_inverter():
            circuit.add_inverter(node_id, node_name)
        elif node_type == Tag.str_node_and2():
            circuit.add_and2(node_id, node_name)
        elif node_type == Tag.str_node_or2():
            circuit.add_or2(node_id, node_name)
        elif node_type == Tag.str_node_xor2():
            circuit.add_xor2(node_id, node_name)
        elif node_type == Tag.str_node_nand2():
            circuit.add_nand2(node_id, node_name)
        elif node_type == Tag.str_node_nor2():
            circuit.add_nor2(node_id, node_name)
        elif node_type == Tag.str_node_xnor2():
            circuit.add_xnor2(node_id, node_name)
        elif node_type == Tag.str_node_maj3():
            circuit.add_maj3(node_id, node_name)
        elif node_type == Tag.str_node_xor3():
            circuit.add_xor3(node_id, node_name)
        elif node_type == Tag.str_node_cell():
            circuit.add_cell(node_id, node_name)
        elif node_type == Tag.str_node_po():
            circuit.add_po(node_id, node_name)
        else:
            raise ValueError(f"Invalid node type: {node_type}")
    
    # add the edges
    for source, target, attr in raw_graph.edges(data=True):
        src = circuit.get_node(source)
        dst = circuit.get_node(target)
        circuit.add_fanin(dst.get_idx(), src.get_idx())
    
    return circuit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a graphml file and return a LogicGraph/CellGraph object')
    parser.add_argument('--file', type=str, help='the name of the file to load')
    parser.add_argument('--type', type=str, help='the type of circuit to load')
    args = parser.parse_args()

    circuit = load_graphml(args.file, args.type)
    print(circuit.to_torch_geometric())