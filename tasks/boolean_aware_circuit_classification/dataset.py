import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
from torch_geometric.data import Data
from typing import List
from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset

class ClassDataset(OpenLS_Dataset):
    """Functional Classification Dataset for the functional classification task

    Args:
        OpenLS_Dataset (_type_): _description_
    """
    def __init__(self, root: str, curr_white_list: List[str], logic:str):
        """_summary_

        Args:
            root (str): dataset root folder
            curr_white_list (list[str]): current white list for the classification
            logic (str): logic type
        """
        super().__init__(root)
        
        self.curr_white_list = curr_white_list
        self.curr_processed_dir = os.path.join(self.processed_dir, logic)
        self.logic = logic
        assert self.logic in self.logics
        assert all(design in self.white_list for design in self.curr_white_list)
        
        self.data_list: List[Data] = self.extract_and_label_subdataset()
        
    def extract_and_label_subdataset(self):
        """_summary_
        Args:
            curr_white_list (list[str]): _description_
            logic (_type_): _description_
        """
        dataset = []

        label = 0
        for design in self.curr_white_list:
            for key, pack in self.data_list:
                if design in key:
                    data = pack[logic]
                    circuit: Circuit = data["circuit"].values[0]
                    graph = circuit.to_torch_geometric()
                    graph.y = torch.tensor(label, dtype=torch.long) # label this graph 
                    dataset.append(graph)
            label += 1
        return dataset
    
    def print_data_list(self):
        for data in self.data_list:
            print("nodes:", data.x)
            print("edges:", data.edge_index)
            print("label:", data.y)
    
if __name__ == "__main__":
    folder:str = sys.argv[1]
    
    curr_white_list = ["i2c"]
    logic = "abc"
    db = ClassDataset(folder, curr_white_list, logic)
    db.print_data_list()
    