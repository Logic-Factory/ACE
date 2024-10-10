import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List, Tuple
from simulate_tt import simulate_tt

import pandas as pd
import numpy as np


from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset
from src.utils.feature import padding_feature_to,padding_feature_to_nochange
from src.io.load_graphml import load_graphml
from src.io.load_qor import QoR, load_qor
from src.io.load_seq import load_seq
from torch_geometric.data import Data



class PP_Data(Data):
  def __init__(self,x=None,x_feature = None,edge_index=None,y=None,label=None,forward_level=None,backward_level=None,forward_index =None,backward_index = None,gate=None,tt= None):
    super().__init__()
    self.x  = x
    self.x_feature = x_feature
    self.edge_index = edge_index
    self.y = y
    self.label  = label
    self.forward_level = forward_level
    self.backward_level= backward_level
    self.forward_index = forward_index
    self.backward_index = backward_index
    self.gate = gate
    self.tt = tt
  
  def __inc__(self,key,value,*args,**kwargs):
    if 'index' in key:
      return self.num_nodes
    else:
      return 0
  
  def __cat_dim__(self,key,value,*args,**kwargs):
    if key in ['forward_index','backward_index']:
      return 0
    elif key in ['edge_index']:
      return  1
    else:
      return 0 
    



class Probability_prediction(OpenLS_Dataset):
    """Functional Classification Dataset for the functional classification task

    Args:
        OpenLS_Dataset (_type_): _description_
    """
    def __init__(self, root: str, recipe_size, curr_white_list: List[str], logic:str, feature_size:int):
        """_summary_

        Args:
            root (str): dataset root folder
            curr_white_list (list[str]): current white list for the classification
            logic (str): logic type
        """
        # self.processed_dir = 'lm_data'
        self.processed_dir = "/data/liumiao/code/OpenLS-D/tasks/Pro_predict/data_pp"
        super().__init__(root, recipe_size)
        
        self.curr_white_list = curr_white_list
        
        self.curr_processed_dir = os.path.join(self.processed_dir, logic)
        self.logic = logic
        self.feature_size = feature_size
        self.count_classes = 0
        assert self.logic in self.logics
        # assert all(design in self.white_list for design in self.curr_white_list)
        
        self.data_list: List[PP_Data] = self.extract_and_label_subdataset()
        
    def extract_and_label_subdataset(self):
        """_summary_
        Args:
            curr_white_list (list[str]): _description_
            logic (_type_): _description_
        """
        data_path =  "data_pp/data.pth"
        if os.path.exists(data_path):
          dataset = torch.load(data_path)
          # print("dataset",dataset)
          # os.system("pause")
          return dataset
        else:
          dataset = []
          self.count_classes = 0
          for design in self.curr_white_list:
              for key, pack in tqdm(self.data_list):
                  if design in key:
                      data = pack[self.logic]
                      # print(data)
                      circuit: Circuit = data["circuit"].values[0]
                      x = circuit.get_node_features()
                      edge_index=circuit.get_edge_index()
                      # graph = circuit.to_torch_geometric()

                      tt = simulate_tt(circuit)
                      y = torch.tensor(self.count_classes, dtype=torch.long)         # label this graph
                      label = torch.tensor(tt,dtype=torch.float32)
                      forward_level,backward_level,forward_index,backward_index = circuit.get_level()
                      
                      gate = circuit.get_gate()
                      x_feature = padding_feature_to_nochange(x, self.feature_size)
                      graph = PP_Data(x,x_feature,edge_index,y,label,forward_level,backward_level,forward_index,backward_index,gate,tt)
                          # padding feature to the same size
                      dataset.append(graph)
                      #######
                      # sys.pause()
                      #######
              self.count_classes += 1
          torch.save(dataset,data_path)

          return dataset

    
    def num_classes(self):
        return self.count_classes
    
    def split_train_test(self, train_ratio: float = 0.8) -> Tuple[List[Data], List[Data]]:
        """Split the dataset into training and testing sets."""
        train_size = int(train_ratio * len(self.data_list))
        shuffled_indices = torch.randperm(len(self.data_list)).tolist()
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]
        train_dataset = [self.data_list[i] for i in train_indices]
        test_dataset = [self.data_list[i] for i in test_indices]
        return train_dataset, test_dataset
        
    def print_data_list(self):
        for data in self.data_list:
            print('forward_index',data.forward_index.shape)
            # print("nodes:", data.x)
            # print("nodes.shape:", data.x.shape)
            print("edges:", data.edge_index.shape)
            print("NOT_mask",data.gate == 4)
            # print("and_mask",data.gate.squeeze(1) == 1)
            
            # print("y:", data.y)
            # print("x_feature:", data.x_feature)
            # print("x_feature:", data.x_feature)
            # print("label:", data.label.shape)
            # print("label:", data.label.shape)
    
if __name__ == "__main__":
    folder:str = sys.argv[1]
    
    curr_white_list = ["i2c"]
    recipe_size = 50
    logic = "abc"
    feature_size = 64
    db = Probability_prediction(folder,recipe_size, curr_white_list, logic, feature_size)
    db.print_data_list()
    