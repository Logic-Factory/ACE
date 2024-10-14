import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


import pandas as pd
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

class RepresentationDataset(Dataset):
    """Functional Classification Dataset for the functional classification task

    Args:
        OpenLS_Dataset (_type_): _description_
    """
    def __init__(self, root_openlsd:str, recipe_size:int, curr_designs:List[str], processed_dir:str):
        """_summary_

        Args:
            root_openlsd (str): root folder of openlsd dataset
            curr_designs (list[str]): current white list for the classification
            logic (str): logic type
        """
        self.root_openlsd:str = os.path.abspath(root_openlsd)
        self.recipe_size:int = int(recipe_size)
        self.curr_designs:List[str] = curr_designs
        self.processed_dir:str = os.path.abspath(processed_dir) # store the processed data_list
        self.logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]
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
    def raw_case_list(self):
        cases = []
        for design in self.curr_designs:
            for i in range( self.recipe_size ):
                conbinations = list(itertools.combinations(self.logics, 2))
                for pairs_logic in conbinations:
                    cases.append( self.design_recipe_logic_pair_name(design, i, pairs_logic[0], pairs_logic[1], "pos") )
                    cases.append( self.design_recipe_logic_pair_name(design, i, pairs_logic[0], pairs_logic[1], "neg") )
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
            print("load circuit representation from pt file")
            self.load_processed_data()
        else:
            print("load circuit representation from openls-d file")
            self.openlsd:Dataset = OpenLS_Dataset(self.root_openlsd, self.recipe_size, self.curr_designs)
            print("load the circuit representation db from openls-d file")
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
            if design not in self.curr_designs:
                continue
            recipes_pack = entry["design_recipes"]
            for i in range(self.recipe_size):
                conbinations = list(itertools.combinations(self.logics, 2))
                for pairs_logic in conbinations:
                    select0 = True  # flag to select the circuit
                    data_0 = recipes_pack[i][pairs_logic[0]]
                    data_1 = recipes_pack[i][pairs_logic[1]]
                    circuit_0: Circuit = data_0["circuit"].values[0]
                    circuit_1: Circuit = data_1["circuit"].values[0]
                    graph_0 = circuit_0.to_torch_geometric()
                    graph_1 = circuit_1.to_torch_geometric()
                    
                    area_0, timing_0, power_0 = data_0["area"].values[0], data_0["timing"].values[0], data_0["power"].values[0]
                    area_1, timing_1, power_1 = data_1["area"].values[0], data_1["timing"].values[0], data_1["power"].values[0]
                    
                    # skip the circuit pair with the same QoR
                    if( float_approximately_equal(timing_0, timing_1) and float_approximately_equal(power_0, power_1) and float_approximately_equal(area_0, area_1) ):
                        continue
                    
                    # tie break for the order
                    if timing_0 > timing_1:
                        select0 = False
                    elif power_0 > power_1:
                        select0 = False
                    elif area_0 > area_1:
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
    folder:str = sys.argv[1]
    recipe_size:int = sys.argv[2]
    target:str = sys.argv[3]
    curr_designs = ["i2c", "priority", "ss_pcm", "tv80"]
    db = RepresentationDataset(root_openlsd=folder, recipe_size=recipe_size, curr_designs=curr_designs, processed_dir=target)