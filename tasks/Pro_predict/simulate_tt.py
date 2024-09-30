import os
import argparse
import sys
import random
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from src.circuit.tag import Tag
from src.operator.compute import *
from src.io.load_graphml import load_graphml
import torch
# python analysis/simulate_tt.py --file dataset/s27_comb/aig/recipe_0.fpga.graphml --type AIG




def simulate_tt(circuit):
  tts = [0 for _ in range(circuit.num_nodes())]
  fanins = [node._fanins for node in circuit._nodes]
  types = [node._type for node in circuit._nodes]
  # print("circuit.num_nodes()",circuit.num_nodes())
  # print(types)
  for _ in range(10000):
    tt = [-1 for _ in range(circuit.num_nodes())]
    random_fanins = [random.randint(0, 1) for _ in range(circuit.num_pis())]
    cnt = 0
    for idx,type in enumerate(types):
      if type == Tag.str_node_const0():
        tt[idx] = False
      if type == Tag.str_node_const1():
        tt[idx] = True
      if type == Tag.str_node_pi():
        tt[idx] = random_fanins[cnt]
        cnt += 1
    for idx,fanin in enumerate(fanins):
      if len(fanin)> 0:
        if circuit.node_at(idx).get_type() == Tag.str_node_and2():
          tt[idx] = sim_and2(bool(fanin[0]),bool(fanin[0]))
          # tt[idx] = tt[fanin[0]] * tt[fanin[1]]
        if circuit.node_at(idx).get_type() == Tag.str_node_inv():
          tt[idx] = 1 - tt[fanin[0]]
        if circuit.node_at(idx).get_type() == Tag.str_node_po():
          tt[idx] = tt[fanin[0]]
    # print("tts",tts)
    tts = [a + int(b) for a,b in zip(tts,tt)]
  tts = [a/10000 for a in tts]
  # print("tts",tts)
  return tts
  return torch.tensor(tts,dtype=torch.float32)

def get_geometric_data(circuit):
    circuit = simulate_tt(circuit)
    data = circuit.to_torch_geometric()
    label =  torch.tensor(circuit._tts,dtype=torch.float32)
    label =label.unsqueeze(1)
    data.label = label
    return data




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a graphml file and return a LogicGraph/CellGraph object')
    parser.add_argument('--file', type=str, help='the name of the file to load')
    parser.add_argument('--type', type=str, help='the type of circuit to load')
    args = parser.parse_args()

    circuit = load_graphml(args.file)
    # print( [node._fanins for node in circuit._nodes])
    data = get_geometric_data(circuit)
    print(data)
    # print(label.shape)
    
    
    
    
    # print(circuit.to_torch_geometric())