import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import networkx as nx
import gzip

from src.circuit.tag import Tag
from src.circuit.node import Node
from src.circuit.circuit import Circuit


def load_graphml(filename:str) -> Circuit:
    """
    Load a graphml file and return a LogicGraph/CellGraph object
    :param filename: the name of the file to load
    :return: a LogicGraph object
    """
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            raw_graph = nx.read_graphml(f)
    else:
        raw_graph = nx.read_graphml(filename)
    circuit = Circuit()
    
    # only add the nodes, leave the fanins alone
    for id, attr in raw_graph.nodes(data=True):
        node_type = attr.get('type')
        node_func = attr.get('func')
        assert node_type in Tag.tags_node(), f"Invalid node type: {node_type}"

        node_id = str(id)
        if node_type in Tag.str_all_const0():
            circuit.add_const0(node_id, node_func)
        elif node_type in Tag.str_node_const1():
            circuit.add_const1(node_id, node_func)
        elif node_type == Tag.str_node_pi():
            circuit.add_pi(node_id, node_func)
        elif node_type == Tag.str_node_po():
            circuit.add_po(node_id, [], node_func)
        # logic gates
        elif node_type == Tag.str_node_inv():
            circuit.add_inverter(node_id, [], node_func)
        elif node_type == Tag.str_node_buf():
            circuit.add_inverter(node_id, [], node_func)
        elif node_type == Tag.str_node_and2():
            circuit.add_and2(node_id, [], node_func)
        elif node_type == Tag.str_node_or2():
            circuit.add_or2(node_id, [], node_func)
        elif node_type == Tag.str_node_xor2():
            circuit.add_xor2(node_id, [], node_func)
        elif node_type == Tag.str_node_nand2():
            circuit.add_nand2(node_id, [], node_func)
        elif node_type == Tag.str_node_nor2():
            circuit.add_nor2(node_id, [], node_func)
        elif node_type == Tag.str_node_xnor2():
            circuit.add_xnor2(node_id, [], node_func)
        elif node_type == Tag.str_node_maj3():
            circuit.add_maj3(node_id, [], node_func)
        elif node_type == Tag.str_node_xor3():
            circuit.add_xor3(node_id, [], node_func)
        elif node_type == Tag.str_node_nand3():
            circuit.add_nand3(node_id, [], node_func)
        elif node_type == Tag.str_node_nor3():
            circuit.add_nor3(node_id, [], node_func)
        elif node_type == Tag.str_node_mux21():
            circuit.add_mux21(node_id, [], node_func)
        elif node_type == Tag.str_node_nmux21():
            circuit.add_nmux21(node_id, [], node_func)
        elif node_type == Tag.str_node_aoi21():
            circuit.add_aoi21(node_id, [], node_func)
        elif node_type == Tag.str_node_oai21():
            circuit.add_oai21(node_id, [], node_func)
        elif node_type == Tag.str_node_axi21():
            circuit.add_axi21(node_id, [], node_func)
        elif node_type == Tag.str_node_xai21():
            circuit.add_xai21(node_id, [], node_func)
        elif node_type == Tag.str_node_oxi21():
            circuit.add_oxi21(node_id, [], node_func)
        elif node_type == Tag.str_node_xoi21():
            circuit.add_xoi21(node_id, [], node_func)
        # standard cell / LUT
        else:
            assert (not node_type.startswith("GTECH_"))
            circuit.add_cell(node_id, [], node_func)
    
    # add the edges
    # for source, target, attr in raw_graph.edges(data=True):
    #     src = circuit.get_node(source)
    #     dst = circuit.get_node(target)
    #     print(src.get_idx(), " -> ", dst.get_idx())
    #     print(src.get_type(), " -> ", dst.get_type())

    for source, target, attr in raw_graph.edges(data=True):
        src = circuit.get_node(source)
        dst = circuit.get_node(target)
        circuit.add_fanin(dst.get_idx(), src.get_idx())
    return circuit

if __name__ == '__main__':
    file = sys.argv[1]
    circuit = load_graphml(file)
    print(circuit.to_torch_geometric())