import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from src.circuit.circuit import Circuit, Node, Tag

def profile_circuit(circuit: Circuit):
    """_summary_

    Args:
        circuit (Circuit): _description_
    """
    pis = circuit.num_pis()
    pos = circuit.num_pos()
    gates = circuit.num_gates()
    edges = circuit.num_edges()
    
    profile = {}
    
    def count_gate_type(node: Node):
        node_type = node.get_type()
        if node_type in profile:
            profile[node_type] += 1
        else:
            profile[node_type] = 1
    
    # count the node type of each cell
    circuit.foreach_gate(count_gate_type)
    
    print("Circuit Profile:")
    print('num pis: ', pis)
    print('num pos: ', pos)
    print('num gates: ', gates)
    print('num edges: ', edges)
    print('type counts: ')
    for key, value in profile.items():
        print("\t", key, ': ', value)

from src.io.load_graphml import load_graphml

if __name__ == '__main__':
    file = sys.argv[1]
    circuit = load_graphml(file)
    profile_circuit(circuit)