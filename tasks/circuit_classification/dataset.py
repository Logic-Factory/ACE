import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset


class ClassificationDataset(Dataset):
    def __init__(self, root_openlsd:str, processed_dir:str, designs:List[str], logic:str, recipes:int):
        """_summary_
        
        Args:
            root_openlsd (str): root folder of openlsd dataset
            processed_dir (str): processed data folder for current task
            designs (list[str]): current white list for current task
            logic (str): logic type
            recipes: (int): recipe size
        """
        super(ClassificationDataset, self).__init__()
        self.root_openlsd:str = os.path.abspath(root_openlsd)
        self.processed_dir:str = os.path.join(processed_dir, logic) # store the processed data_list
        self.designs:List[str] = designs
        self.logic:str = logic
        self.recipes:int = int(recipes)
    
        # store the items of this current task
        self.data_list = []
                        
        os.makedirs(self.processed_dir, exist_ok=True)    
        self.load_data()

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
            processed_files.append(path_pt)
        return processed_files
    
    @property
    def processed_data_exist(self):
        return all(os.path.exists(path) for path in self.processed_data_list)
    
    def load_data(self):
        if self.processed_data_exist:
            print("load circuit classification from pt file")
            self.load_processed_data()
        else:
            print("load openls-d file first")
            self.openlsd:Dataset = OpenLS_Dataset(root=self.root_openlsd, designs=self.designs, logics=[self.logic])
            print("load the circuit classification db from openls-d file")
            self.load_adaptive_subdataset()

    def load_processed_data(self):
        processed_data = self.processed_data_list
        def load_processed_one_design_recipe(one_design_recipe):
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
        for entry in self.openlsd:
            design = entry["design_name"]
            if design not in self.designs:
                continue
            recipes_pack = entry["design_recipes"]
            label_int = self.designs.index(design)
            for i in tqdm(range(self.recipes), desc=f"process at {design}"):
                data = recipes_pack[self.logic][i]
                circuit: Circuit = data["circuit"].values[0]
                graph = circuit.to_torch_geometric()
                graph.y = torch.tensor(label_int, dtype=torch.long)       # label this graph
                self.data_list.append(graph)
                path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
                torch.save(graph, path_pt)
    
    def num_classes(self):
        return len(self.designs)
    
    def split_train_test(self, train_ratio: float = 0.8) -> Tuple[List[Data], List[Data]]:
        """Split the dataset into training and testing sets."""
        train_size = int(train_ratio * len(self.data_list))
        shuffled_indices = torch.randperm(len(self.data_list)).tolist()
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]
        train_dataset = [self.data_list[i] for i in train_indices]
        test_dataset = [self.data_list[i] for i in test_indices]
        return train_dataset, test_dataset

if __name__ == "__main__":
    folder:str = sys.argv[1]
    recipe_size:int = sys.argv[2]
    target:str = sys.argv[3]
    curr_designs = [
        "ctrl",
        "steppermotordrive",
        "router",
        "int2float",
        "ss_pcm",
        "usb_phy",
        "sasc",
        ]
    db = ClassificationDataset(root_openlsd=folder, recipe_size=recipe_size, curr_designs=curr_designs, processed_dir=target, logic="abc")