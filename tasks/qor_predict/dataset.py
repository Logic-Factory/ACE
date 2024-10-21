import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)


from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import json
from copy import deepcopy
import json
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from src.io.load_graphml import load_graphml
from src.dataset.dataset import OpenLS_Dataset

def cmd_to_number(seq):
    cmd_to_num = {
        "refactor": 1,
        "refactor -l": 2,
        "refactor -z": 3,
        "refactor -l -z": 4,
        "rewrite": 5,
        "rewrite -l": 6,
        "rewrite -z": 7,
        "rewrite -l -z": 8,
        "resub": 9,
        "resub -l": 10,
        "resub -z": 11,
        "resub -l -z": 12,
        "balance": 13
    }
    seq_str = seq.iloc[0]
    commands = seq_str.split(';')
    number_seq = [cmd_to_num.get(cmd.strip(), -1) for cmd in commands if cmd.strip() in cmd_to_num]
    while len(number_seq) < 20:
        number_seq.append(0)
    return number_seq

class QoR_Dataset(Dataset):
    """Functional QoR predict Dataset for the functional QoR predict task

    Args:
        OpenLS_Dataset (_type_): _description_
    """
       
    def __init__(self, root_openlsd:str, recipe_size:int, curr_designs:List[str], processed_dir:str,logic:str, target:str='area'):
        """_summary_

        Args:l
            root_openlsd (str): root folder of openlsd dataset
            recipe_size (int): number of recipes for each design
            curr_designs (list[str]): current white list for the QoR predict
            precessed_dir (str): processed data_list
            logic (str): logic type
            target (str, optional): target QoR. Defaults to 'area' or 'delay'.
        """
        self.root_openlsd:str = os.path.abspath(root_openlsd)
        self.recipe_size:int = int(recipe_size)
        self.curr_designs:List[str] = curr_designs
        self.processed_dir:str = os.path.join(processed_dir, logic) # store the processed data_list
        self.logic:str = logic
        self.data_list = []
        self.target = target

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
        for design in self.curr_designs:
            for i in range( self.recipe_size ):
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
            print("load qor prediction with sequence from pt file")
            self.load_processed_data()
        else:
            print("load qor prediction with sequence from openls-d file")
            self.openlsd:Dataset = OpenLS_Dataset(self.root_openlsd, self.recipe_size, self.curr_designs)
            print("load the qor prediction with sequence dp from openls-d file")
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
        design_data = {}
        for entry in self.openlsd:
            datas = []
            design = entry["design_name"]
            if design not in self.curr_designs:
                continue
            recipes_pack = entry["design_recipes"]
            label_int = self.curr_designs.index(design)
            graph_dir = os.path.join(self.root_openlsd, design,'raw.gtech.aig.graphml')
            for i in range(self.recipe_size):
                if self.target == 'area':
                    datas.append(recipes_pack[i][self.logic]['area'])
                elif self.target == 'delay':
                    datas.append(recipes_pack[i][self.logic]['delay'])
                else:
                    raise ValueError("target must be 'area' or 'delay'")
            mean_, std_ = np.mean(datas), np.std(datas)
            design_data[design] = {'mean': mean_, 'std': std_}

            circuit = load_graphml(graph_dir)
            graph = circuit.to_torch_geometric()
            
            for i in range(self.recipe_size):
                if self.target == 'area':
                    value = (recipes_pack[i][self.logic]['area'] - mean_)/std_
                elif self.target == 'delay':
                    value = (recipes_pack[i][self.logic]['delay'] - mean_)/std_
                else:
                    raise ValueError("target must be 'area' or 'delay'")
                graph_tmp = deepcopy(graph)
                # print(recipes_pack[i][self.logic]['seq'])
                seq = cmd_to_number(recipes_pack[i][self.logic]['seq'])
                graph_tmp.name = design
                graph_tmp.seq = torch.tensor(seq,dtype=torch.long)
                graph_tmp.target = torch.tensor(value,dtype=torch.float32)
                self.data_list.append(graph_tmp)
                path_pt = os.path.join(self.processed_dir, f"{self.design_recipe_name(design, i)}.pt")
                torch.save(graph_tmp,path_pt)
        json_dir = os.path.join(self.root_openlsd, 'design_data.json')
        print(json_dir)
        with open(json_dir, 'w') as f:
            json.dump(design_data, f, indent=4)
            
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
            print("seq:", data.seq)
            print("target:", data.target)
            print("name:", data.name)

if __name__ == "__main__":
    folder:str = sys.argv[1]
    recipe_size:int = sys.argv[2]
    target:str = sys.argv[3]
    curr_designs = ["i2c"]
    processed_dir = sys.argv[4]
    logic = "abc"

    db = QoR_Dataset(root_openlsd=folder, recipe_size=recipe_size, curr_designs=curr_designs, processed_dir=processed_dir,logic =logic, target =target)