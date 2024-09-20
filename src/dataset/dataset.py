import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import re
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from src.circuit.circuit import Circuit, Node, Tag
from src.io.load_graphml import load_graphml
from src.io.load_qor import QoR, load_qor
from src.io.load_seq import load_seq, load_raw_seq

def sort_by_recipe_number(file_path):
    match = re.search(r'recipe_(\d+)', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    else:
        return float('inf')

class OpenLS_Dataset(Dataset):
    def __init__(self, root:str, recipe_size:int = 500, transform=None, pre_transform=None):
        """_summary_

        Args:
            root (str): _description_
            recipe_size (int, optional): _description_. Defaults to 500. 500 is the default size of the dataset for each design.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        self.root = os.path.abspath(root)
        self.processed_dir = os.path.join(self.root, "processed_dir")
        self.recipe_size = recipe_size
        self.logics = ["abc", "aig", "oig", "xag", "primary", "mig", "gtg"]
        self.white_list = ["i2c", "fir"]
        self.data_list = []
        self.transform = transform
        
        super().__init__()
        os.makedirs(self.processed_dir, exist_ok=True)
        self.load_data()
        
    def __len___(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def str_case_name(self, design, recipe_number):
        return f"{design}_recipe_{recipe_number}"

    @property
    def raw_case_list(self):
        cases = []
        for design in self.white_list:
            for i in range(self.recipe_size):
                case_name =  self.str_case_name(design, i)
                cases.append(case_name)
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
        file_paths = [os.path.join(self.processed_dir, fname) for fname in self.processed_data_list]
        return all(os.path.exists(path) for path in file_paths)
    
    def load_processed_data(self):
        processed_data = self.processed_data_list
        for case in tqdm(processed_data, desc="waiting"):
            basename = os.path.basename(case)
            filename = os.path.splitext(basename)[0]
            path = os.path.join(self.processed_dir, case)
            data = torch.load(path, weights_only=False)
            self.data_list.append( [filename, data] )

    def load_data(self):
        if self.processed_data_exist:
            print("load from pt file")
            self.load_processed_data()
        else:
            print("load from source file")
            for design in os.listdir(self.root):
                if design not in self.white_list:
                    continue
                print("load at: ", design)
                for i in tqdm( range(self.recipe_size), desc="waiting"):
                    self.load_one_logic(self.root, design, i)
    
    def load_one_logic(self, folder, design, index):
        pack = {}
        key = self.str_case_name(design, index)
        path_one_design = os.path.join(folder, design)
        for logic in self.logics:
            logic_file = os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml")
            area_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json")
            timing_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json")
            power_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json")
            seq_file = os.path.join(path_one_design, logic, f"recipe_{index}.seq")
            if not os.path.exists(logic_file) or not os.path.exists(area_file) or not os.path.exists(timing_file) or not os.path.exists(power_file):
                print("design recipe not complete, and skip this: ", key)
                continue
            
            circuit = load_graphml(logic_file)
            area = load_qor(area_file).get_area()
            timing = load_qor(timing_file).get_timing()
            power = load_qor(power_file).get_power_total()
            
            seq = ""
            if logic == "abc":
                seq = load_raw_seq(seq_file)
            data = pd.DataFrame({
                "circuit": [circuit],
                "type": [logic],
                "seq": [seq],
                "area": [area],
                "timing": [timing],
                "power": [power]
            })
            pack[logic] = data
        
        if pack:
            # print("pack: ", pack )
            self.data_list.append( [key, pack] )
            path_pt = os.path.join(self.processed_dir, self.str_case_name(design, index) + ".pt")
            torch.save(pack, path_pt)
    
    def print_data_list(self):
        print("data list size", len(self.data_list))
        for key, pack in self.data_list:
            print("key: ", key)
            for logic in self.logics:
                data = pack[logic]
                print("type: ", data["type"].values[0])
                print("area: ", data["area"].values[0])
                print("timing: ", data["timing"].values[0])
                print("power: ", data["power"].values[0])
                print("seq: ", data["seq"].values[0])
                circuit: Circuit = data["circuit"].values[0]
                print("pis: ", circuit.num_pis())
                print("pos: ", circuit.num_pos())
                print("gates: ", circuit.num_gates())
                print("edges: ", circuit.num_edges())
        

if __name__ == "__main__":
    folder:str = sys.argv[1]
    db = OpenLS_Dataset(folder)
    db.print_data_list()