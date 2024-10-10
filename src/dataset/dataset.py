import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import re
import glob
import gzip
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from src.circuit.circuit import Circuit, Node, Tag
from src.io.load_graphml import load_graphml
from src.io.load_qor import QoR, load_qor
from src.io.load_seq import load_seq

class OpenLS_Dataset(Dataset):
    def __init__(self, root:str, recipe_size:int = 1000, design_list:List[str] = []):
        """_summary_

        Args:
            root (str): _description_
            recipe_size (int, optional): _description_. Defaults to 500. 500 is the default size of the dataset for each design.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
        """
        self.root = os.path.abspath(root)
        self.processed_dir = os.path.join(self.root, "processed_dir")
        self.recipe_size = int(recipe_size)
        self.logics = ["abc", "aig", "oig", "xag", "primary", "mig", "gtg"]
        
        self.design_list = design_list       # store the used designs
        if design_list == []:
            self.design_list = self.raw_all_case_list()
        else:
            assert all([design in self.raw_all_case_list() for design in design_list]), "design not in the dataset"

        super().__init__()
                
        self.data_list = []
        os.makedirs(self.processed_dir, exist_ok=True)
        self.load_data()
        
    def __len___(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def raw_all_case_list(self):
        cases = []
        for design in os.listdir(self.root):
            if design == "processed_dir":
                continue
            cases.append(design)
        return cases

    @property
    def processed_data_list(self):
        cases = self.design_list
        processed_files = []
        for i in range(len(cases)):
            path_pt = os.path.join(self.processed_dir, f"{cases[i]}_pack_{self.recipe_size}.pt")
            processed_files.append(path_pt)
        return processed_files
    
    @property
    def processed_data_exist(self):
        return all(os.path.exists(path) for path in self.processed_data_list)
    
    def load_data(self):
        if self.processed_data_exist:
            print("load openls-d from pt file")
            self.load_processed_data()
        else:
            print("load openls-d from source file")
            cases = self.design_list
            for design in tqdm( cases, desc="loading"):
                self.load_one_design(self.root, design)
        # sort the data_list according the desing_name
        self.data_list.sort(key=lambda x: x["design_name"])
        
    def load_processed_data(self):
        processed_data = self.processed_data_list
        
        def load_processed_design(design):
            data = torch.load(design, weights_only=False)
            self.data_list.append(data)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for design in tqdm(processed_data, desc="loading"):
                futures.append(executor.submit(load_processed_design, design))
            for future in as_completed(futures):
                future.result()
    
    def load_one_design(self, folder, desgin):
        recipe_list = []
        path_one_design = os.path.join(folder, desgin)
        
        raw_aig_file = ""
        raw_gtech_file = ""
        if os.path.exists( os.path.join(path_one_design, f"raw.gtech.aig.graphml") ):
            raw_aig_file = os.path.join(path_one_design, f"raw.gtech.aig.graphml")
        elif os.path.exists( os.path.join(path_one_design, f"raw.gtech.aig.graphml.zst") ):
            raw_aig_file = os.path.join(path_one_design, f"raw.gtech.aig.graphml.zst")
        elif os.path.exists( os.path.join(path_one_design, f"raw.gtech.aig.graphml.gz") ):
            raw_aig_file = os.path.join(path_one_design, f"raw.gtech.aig.graphml.gz")
        else:
            print("no raw aig file")
            assert False

        if os.path.exists( os.path.join(path_one_design, f"raw.gtech.graphml") ):
            raw_gtech_file = os.path.join(path_one_design, f"raw.gtech.graphml")
        elif os.path.exists( os.path.join(path_one_design, f"raw.gtech.graphml.zst") ):
            raw_gtech_file = os.path.join(path_one_design, f"raw.gtech.graphml.zst")
        elif os.path.exists( os.path.join(path_one_design, f"raw.gtech.graphml.gz") ):
            raw_gtech_file = os.path.join(path_one_design, f"raw.gtech.graphml.gz")
        else:
            print("no raw gtech file")
            assert False
        
        src_aig = load_graphml(raw_aig_file)
        src_gtech = load_graphml(raw_gtech_file)
        
        # load the recipes parallelly
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for i in range(self.recipe_size):
                futures.append( executor.submit(self.load_one_logic, folder, desgin, i) )
            recipe_list = [future.result() for future in futures]   # preserve the original order of the recipe index
        
        data = {
            "design_name": desgin,
            "design_gtech": src_gtech,
            "design_aig": src_aig,
            "design_recipes": recipe_list
        }
        self.data_list.append( data )
        path_pt = os.path.join(self.processed_dir, f"{desgin}_pack_{self.recipe_size}.pt")
        torch.save(data, path_pt)
            
    def load_one_logic(self, folder, design, index):
        pack = {}
        path_one_design = os.path.join(folder, design)
        for logic in self.logics:
            logic_file = ""
            area_file = ""
            timing_file = ""
            power_file = ""
            seq_file = ""
            
            if os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml") ):
                logic_file = os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml.zst") ):
                logic_file = os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml.zst")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml.gz") ):
                logic_file = os.path.join(path_one_design, logic, f"recipe_{index}.logic.graphml.gz")
            else:
                print("no logic file")
            
            if os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json") ):
                area_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json.zst") ):
                area_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json.zst")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json.gz") ):
                area_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.qor.json.gz")
            else:
                print("no area file")

            if os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json") ):
                timing_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json.zst") ):
                timing_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json.zst")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json.gz") ):
                timing_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.timing.qor.json.gz")
            else:
                print("no timing file")

            if os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json") ):
                power_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json.zst") ):
                power_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json.zst")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json.gz") ):
                power_file = os.path.join(path_one_design, logic, f"recipe_{index}.asic.power.qor.json.gz")
            else:
                print("no power file")
            
            if os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.seq") ):
                seq_file = os.path.join(path_one_design, logic, f"recipe_{index}.seq")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.seq.zst") ):
                seq_file = os.path.join(path_one_design, logic, f"recipe_{index}.seq.zst")
            elif os.path.exists( os.path.join(path_one_design, logic, f"recipe_{index}.seq.gz") ):
                seq_file = os.path.join(path_one_design, logic, f"recipe_{index}.seq.gz")
            else:
                if logic == "abc":
                    print("no seq file")

            if logic_file == "" or area_file == "" or timing_file == "" or power_file == "":
                print("design recipe not complete, and skip this: ", f"{design}_recipe_{index}")
                continue

            circuit = load_graphml(logic_file)
            area = load_qor(area_file).get_area()
            timing = load_qor(timing_file).get_timing()
            power = load_qor(power_file).get_power_total()
            
            seq = ""
            if logic == "abc":
                seq = load_seq(seq_file)
            data = pd.DataFrame({
                "circuit": [circuit],
                "type": [logic],
                "seq": [seq],
                "area": [area],
                "timing": [timing],
                "power": [power]
            })
            pack[logic] = data
        return pack

if __name__ == "__main__":
    folder:str = sys.argv[1]
    recipe_size:int = sys.argv[2]
    design_list = ["i2c", "priority", "ss_pcm", "tv80"]
    # design_list = []
    db = OpenLS_Dataset(folder, recipe_size, design_list)