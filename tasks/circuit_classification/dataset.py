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
from torch_geometric.utils import from_networkx, to_undirected
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

    def split_train_test(self, designs, train_ratio: float = 0.8, is_directed = True):
        train_dataset = []
        test_dataset = []
        
        # relable the dataset
        design_to_label = {design: i for i, design in enumerate( designs )}
        # collect all the data with explicit label
        label_datas_dict = {}
        for data in self.data_list:
            label = data.y.item()
            if label not in label_datas_dict:
                label_datas_dict[label] = []
            label_datas_dict[label].append(data)
        
        for design in designs:
            # label = self.label_dict[design]
            label = design_to_label[design]
            datas = label_datas_dict[label]

            for data in datas:
                data.y = torch.tensor([label], dtype=torch.long)
                
                if is_directed is False:
                    data.edge_index = to_undirected(data.edge_index)
            
            train_size = int(len(datas) * train_ratio)
            train = datas[:train_size]
            test = datas[train_size:]
            train_dataset.extend(train)
            test_dataset.extend(test)
        
        # print the count of each label for the train and test dataset
        train_label_count = {}
        test_label_count = {}
        for data in train_dataset:
            label = data.y.item()
            if label not in train_label_count:
                train_label_count[label] = 0
            train_label_count[label] += 1
        for data in test_dataset:
            label = data.y.item()
            if label not in test_label_count:
                test_label_count[label] = 0
            test_label_count[label] += 1
        print("train label count:", train_label_count)
        print("test label count:", test_label_count)
        
        return train_dataset, test_dataset

if __name__ == "__main__":
    folder:str = sys.argv[1]
    recipe_size:int = sys.argv[2]
    target:str = sys.argv[3]
    curr_designs = [
        "router",
        "usb_phy",
        "cavlc",
        "adder",
        "systemcdes",
        "max",
        "spi",
        "wb_dma",
        "des3_area",
        "tv80",
        "arbiter",
        "mem_ctrl",
        "square",
        "aes",
        "fpu",
        ]
    db = ClassificationDataset(root_openlsd=folder, recipe_size=recipe_size, curr_designs=curr_designs, processed_dir=target, logic="abc")