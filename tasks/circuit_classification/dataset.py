import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
from torch_geometric.data import Data
from typing import List, Tuple

from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset
from src.utils.feature import padding_feature_to

class ClassificationDataset(OpenLS_Dataset):
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
        super().__init__(root, recipe_size)
        
        self.curr_white_list = curr_white_list
        self.curr_processed_dir = os.path.join(self.processed_dir, logic)
        self.logic = logic
        self.feature_size = feature_size
        self.count_classes = 0
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

        self.count_classes = 0
        for design in self.curr_white_list:
            for key, pack in self.data_list:
                if design in key:
                    data = pack[self.logic]
                    circuit: Circuit = data["circuit"].values[0]
                    graph = circuit.to_torch_geometric()
                    graph.y = torch.tensor(self.count_classes, dtype=torch.long)         # label this graph
                    graph = padding_feature_to(graph, self.feature_size)    # padding feature to the same size
                    dataset.append(graph)
            self.count_classes += 1
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
            print("nodes:", data.x)
            print("edges:", data.edge_index)
            print("label:", data.y)
    
if __name__ == "__main__":
    folder:str = sys.argv[1]
    
    curr_white_list = ["i2c"]
    logic = "abc"
    feature_size = 64
    db = ClassificationDataset(folder, curr_white_list, logic, feature_size)
    db.print_data_list()
    