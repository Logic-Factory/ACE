import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
from tqdm import tqdm

from typing import List, Tuple
from simulate_tt import simulate_tt

import pandas as pd
import numpy as np
import itertools
import multiprocessing
from tqdm import tqdm

from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset
from src.utils.feature import padding_feature_to_nochange
from src.io.load_graphml import load_graphml


from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.dataset.dataset import OpenLS_Dataset
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from src.utils.numeric import float_approximately_equal


      

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
    



class Probability_prediction(Dataset):
    """Functional Classification Dataset for the functional classification task

    Args:
        OpenLS_Dataset (_type_): _description_
    """
    def __init__(self, root_openlsd,processed_dir,designs,logic, recipes):
        """_summary_

        Args:l
            root_openlsd (str): root folder of openlsd dataset
            designs (list[str]): current white list for the classification
            logic (str): logic type
        """
        self.root_openlsd:str = os.path.abspath(root_openlsd)
        self.recipes:int = int(recipes)
        self.designs:List[str] = designs
        self.processed_dir:str = os.path.abspath(processed_dir) # store the processed data_list
        self.feature_size = 64
        # self.logic = ["aig", "oig", "xag", "primary", "mig", "gtg"]
        self.logic = 'aig'
        self.logic = "aig"
        self.data_list = []
                        
        os.makedirs(self.processed_dir, exist_ok=True)    
        self.load_data()
        
        # self.data_list: List[PP_Data] = self.extract_and_label_subdataset()

    def __len___(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def design_recipe_name(self, design:str, recipe:int):
        return f"{design}_recipe_{recipe}"
    
    @property
    def raw_case_list(self):
        cases = []
        for design in self.designs:
            for i in range( self.recipes ):
                cases.append(f"{self.design_recipe_name(design, i)}")
        return cases
    
    @property
    def processed_data_list(self):
        cases = self.raw_case_list
        processed_files = []
        for i in range(len(cases)):
            path_pt = os.path.join(self.processed_dir, f"{cases[i]}.pt")
            if os.path.exists(path_pt):
                processed_files.append(path_pt)
        return processed_files
    
    @property
    def processed_data_exist(self):
        return all(os.path.exists(path) for path in self.processed_data_list)
    
    def load_data(self):
        # if self.processed_data_exist:
        if True:
            print("load circuit representation from pt file")
            self.load_processed_data()
        else:
            print("load circuit representation from openls-d file")
            self.openlsd:Dataset = OpenLS_Dataset(self.root_openlsd, designs=self.designs, logics=[self.logic])
            print("load the circuit representation db from openls-d file")
            self.load_adaptive_subdataset()

    def load_processed_data(self):
        processed_data = self.processed_data_list
        print('len(processed_data)',len(processed_data))
        def load_processed_one_design_recipe(one_design_recipe):
            if not os.path.exists(one_design_recipe):
                return
            data = torch.load(one_design_recipe, weights_only=False)
            self.data_list.append(data)        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for one_design_recipe in tqdm(processed_data, desc="loading"):
                futures.append(executor.submit(load_processed_one_design_recipe, one_design_recipe))
            for future in futures:
                future.result()





    def load_adaptive_subdataset(self):
        """load adaptive subdataset from openls-d: extract and label
        """
        count = 0
        
        def load_adaptive_subdataset_pool(entry):
            # print("entry",entry)
            design = entry["design_name"]
            if design not in self.designs:
                return
            recipes_pack = entry["design_recipes"]
            label_int = self.designs.index(design)
            for i in tqdm(range(self.recipes), desc=f"process at {design}"):
                path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
                if os.path.exists(path_pt):
                  continue
                data = recipes_pack[self.logic][i]
                circuit: Circuit = data["circuit"].values[0]
                graph = circuit.to_torch_geometric()
                x = circuit.get_node_features()
                edge_index=circuit.get_edge_index()
                    # graph = circuit.to_torch_geometric()

                tt = simulate_tt(circuit)
                # print("++++++++++++")
                y = torch.tensor(label_int, dtype=torch.long)         # label this graph
                label = torch.tensor(tt,dtype=torch.float32)
                forward_level,backward_level,forward_index,backward_index = circuit.get_level()
                gate = circuit.get_gate()
                x_feature = padding_feature_to_nochange(x, self.feature_size)
                graph = PP_Data(x,x_feature,edge_index,y,label,forward_level,backward_level,forward_index,backward_index,gate,tt)

                    # padding feature to the same size
                self.data_list.append(graph)
                path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
                torch.save(graph, path_pt)
  
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for entry in self.openlsd:
                futures.append(executor.submit(load_adaptive_subdataset_pool, entry))
        # for entry in self.openlsd:
        #     # print("entry",entry)
        #     design = entry["design_name"]
        #     if design not in self.designs:
        #         continue
        #     recipes_pack = entry["design_recipes"]
        #     label_int = self.designs.index(design)
        #     for i in tqdm(range(self.recipes), desc=f"process at {design}"):
        #         path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
        #         if os.path.exists(path_pt):
        #           continue
        #         data = recipes_pack[self.logic][i]
        #         circuit: Circuit = data["circuit"].values[0]
        #         graph = circuit.to_torch_geometric()
        #         x = circuit.get_node_features()
        #         edge_index=circuit.get_edge_index()
        #             # graph = circuit.to_torch_geometric()

        #         tt = simulate_tt(circuit)
        #         # print("++++++++++++")
        #         y = torch.tensor(label_int, dtype=torch.long)         # label this graph
        #         label = torch.tensor(tt,dtype=torch.float32)
        #         forward_level,backward_level,forward_index,backward_index = circuit.get_level()
        #         gate = circuit.get_gate()
        #         x_feature = padding_feature_to_nochange(x, self.feature_size)
        #         graph = PP_Data(x,x_feature,edge_index,y,label,forward_level,backward_level,forward_index,backward_index,gate,tt)

        #             # padding feature to the same size
        #         self.data_list.append(graph)
        #         path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
        #         torch.save(graph, path_pt)
  

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



    # @staticmethod
    # def pool_process(self,i,recipes_pack,label_int,design):
    #   data = recipes_pack[self.logic][i]
    #   circuit: Circuit = data["circuit"].values[0]
    #   graph = circuit.to_torch_geometric()
    #   x = circuit.get_node_features()
    #   edge_index=circuit.get_edge_index()
    #       # graph = circuit.to_torch_geometric()
    #   print(count)
    #   count += 1
    #   tt = simulate_tt(circuit)

    #   y = torch.tensor(label_int, dtype=torch.long)         # label this graph
    #   label = torch.tensor(tt,dtype=torch.float32)
    #   forward_level,backward_level,forward_index,backward_index = circuit.get_level()
    #   gate = circuit.get_gate()
    #   x_feature = padding_feature_to_nochange(x, self.feature_size)
    #   graph = PP_Data(x,x_feature,edge_index,y,label,forward_level,backward_level,forward_index,backward_index,gate,tt)
    #       # padding feature to the same size
    #   self.data_list.append(graph)
    #   path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
    #   torch.save(graph, path_pt)
        
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
        print("self.data_list",len(self.data_list))
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
    folder:str = '/data/project_share/openlsd_1028'
    recipes:int = 1000
    target:str = 'data_pp'
    designs = [
        "ctrl",
        "steppermotordrive",
        "router",
        "int2float",
        "ss_pcm",
        "usb_phy",
        "sasc",
        "cavlc",
        "simple_spi",
        "priority"
    ]
    db = Probability_prediction(root_openlsd=folder,processed_dir=target,designs=designs,logic='abc', recipes=recipes)
    db.print_data_list()