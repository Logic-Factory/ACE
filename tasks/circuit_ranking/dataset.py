import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import pandas as pd
from glob import glob
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import itertools

from src.circuit.circuit import Circuit
from src.dataset.dataset import OpenLS_Dataset
from src.utils.numeric import float_approximately_equal

class RankingDataset(Dataset):
    def __init__(self, root_openlsd:str, processed_dir:str, designs:List[str], logics:List[str], recipes:int):
        """_summary_
        
        Args:
            root_openlsd (str): root folder of openlsd dataset
            processed_dir (str): processed data folder for current task
            designs (list[str]): current white list for current task
            logic (str): logic type
            recipes: (int): recipe size
        """
        super(RankingDataset, self).__init__()
        self.root_openlsd:str = os.path.abspath(root_openlsd)
        self.processed_dir:str = os.path.abspath(processed_dir) # store the processed data_list
        self.designs:List[str] = designs
        self.logics = logics
        self.recipes:int = int(recipes)
        
        # store the items of this current task
        self.data_list = []
                        
        os.makedirs(self.processed_dir, exist_ok=True)    
        self.load_data()

    def __len___(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def design_recipe_logic_pair_name(self, design:str, recipe:int, logic_0, logic_1, polar:str):
        return f"{design}_recipe_{recipe}_{logic_0}_{logic_1}_{polar}"
        
    @property
    def processed_data_list(self):
        pattern = os.path.join(self.processed_dir, f'*.pt')
        processed_files = [file for file in glob(pattern, recursive=True)]
        return processed_files
    
    @property
    def processed_data_exist(self):
        return len(self.processed_data_list) > 0 and all(os.path.exists(path) for path in self.processed_data_list)
    
    def load_data(self):
        if self.processed_data_exist:
            print("load circuit ranking from pt file")
            self.load_processed_data()
        else:
            print("load openls-d file first")
            self.openlsd:Dataset = OpenLS_Dataset(root=self.root_openlsd, designs=self.designs, logics=self.logics, recipes=self.recipes)
            print("load the circuit ranking db from openls-d file")
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
            for i in range(self.recipes):
                conbinations = list(itertools.combinations(self.logics, 2))
                for pairs_logic in conbinations:
                    select0 = True  # flag to select the circuit
                    data_0 = recipes_pack[pairs_logic[0]][i]
                    data_1 = recipes_pack[pairs_logic[1]][i]
                    circuit_0: Circuit = data_0["circuit"].values[0]
                    circuit_1: Circuit = data_1["circuit"].values[0]
                    graph_0 = circuit_0.to_torch_geometric()
                    graph_1 = circuit_1.to_torch_geometric()
                    
                    area_0, timing_0= data_0["area"].values[0], data_0["timing"].values[0]
                    area_1, timing_1= data_1["area"].values[0], data_1["timing"].values[0]
                    
                    # skip the circuit pair with the same QoR
                    if( float_approximately_equal(timing_0, timing_1) and float_approximately_equal(area_0, area_1) ):
                        continue
                    
                    # tie break for the order
                    if timing_0 > timing_1:
                        select0 = False
                    else:
                        if area_0 > area_1:
                            select0 = False

                    if select0:
                        graph_0.y = torch.tensor(1, dtype=torch.float)
                        graph_1.y = torch.tensor(0, dtype=torch.float)
                    else:
                        graph_0.y = torch.tensor(0, dtype=torch.float)
                        graph_1.y = torch.tensor(1, dtype=torch.float)
                    pair_name_pos = self.design_recipe_logic_pair_name(design, i, pairs_logic[0], pairs_logic[1], "pos")
                    pair_name_neg = self.design_recipe_logic_pair_name(design, i, pairs_logic[0], pairs_logic[1], "neg")
                    data_pos = {
                        "design": design,
                        "recipe": i,
                        "logic_0": pairs_logic[0],
                        "logic_1": pairs_logic[1],
                        "graph_0": graph_0,
                        "graph_1": graph_1,
                    }
                    data_neg = {
                        "design": design,
                        "recipe": i,
                        "logic_0": pairs_logic[1],
                        "logic_1": pairs_logic[0],
                        "graph_0": graph_1,
                        "graph_1": graph_0,
                    }
                    self.data_list.append(data_pos)
                    self.data_list.append(data_neg)
                    path_pt = os.path.join(self.processed_dir, f"{pair_name_pos}.pt")
                    path_pt = os.path.join(self.processed_dir, f"{pair_name_neg}.pt")
                    torch.save(data_pos, path_pt)
                    torch.save(data_neg, path_pt)
    
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
    root_openlsd:str = sys.argv[1]
    processed_dir:str = sys.argv[2]
    recipes:int = sys.argv[3]
    curr_logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]
    curr_designs = [
        "ctrl",
        "steppermotordrive"
        ]
    db = RankingDataset(root_openlsd=root_openlsd, processed_dir=processed_dir, designs=curr_designs, logics=curr_logics, recipes=recipes)