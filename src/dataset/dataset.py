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
    def __init__(self, root:str, recipe_size:int = 1000, logics:List[str] = [], design_list:List[str] = []):
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
        self.logics = logics
        
        self.design_list = design_list       # store the used designs
        if design_list == []:
            self.design_list = self.raw_all_case_list()
        else:
            # assert all([design in self.raw_all_case_list() for design in design_list]), f"design {design} not in the dataset"
            for design in design_list:
                if design not in self.raw_all_case_list():
                    raise ValueError(f"design {design} not in the dataset")

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
        cases = []
        for design in self.design_list:
            if not os.path.isdir(os.path.join(self.root, design)):
                continue
            cases.append(design)

        processed_files = []
        for i in range(len(cases)):
            case_files = []
            case_raw_pt = os.path.join(self.processed_dir, f"{cases[i]}_raw_{self.recipe_size}.pt")
            case_files.append(case_raw_pt)
            for logic in self.logics:
                case_logic_pt = os.path.join(self.processed_dir, f"{cases[i]}_{logic}_{self.recipe_size}.pt")
                case_files.append(case_logic_pt)
            processed_files.append(case_files)
        return processed_files
    
    @property
    def processed_data_exist(self):
        for case_files in self.processed_data_list:
            if not all(os.path.exists(path) for path in case_files):
                return False
        return True
    
    def load_data(self):
        if self.processed_data_exist:
            print("load openls-d from pt file")
            self.load_processed_data()
        else:
            print("load openls-d from source file")
            cases = self.design_list
            count = 1
            for design in cases:
                if not os.path.isdir(os.path.join(self.root, design)):
                    continue
                print(f"loading {count}/{len(cases)}: {design}")
                count += 1
                self.load_one_design(self.root, design)
        # sort the data_list according the desing_name
        self.data_list.sort(key=lambda x: x["design_name"])
        
    def load_processed_data(self):
        processed_data = self.processed_data_list
        
        def load_processed_design(case_files):
            data_raw = torch.load(case_files[0], weights_only=False)
            recipe_list = []
            for i in range(1, len(case_files)):
                data_logic = torch.load(case_files[i], weights_only=False)
                recipe_list.append(data_logic)
            data = {
                "design_name": data_raw["design_name"],
                "design_gtech": data_raw["design_gtech"],
                "design_aig": data_raw["design_aig"],
                "design_recipes": recipe_list
            }
            self.data_list.append(data)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for case_files in tqdm(processed_data, desc="loading"):
                futures.append(executor.submit(load_processed_design, case_files))
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
        
        data_raw ={
            "design_name": desgin,
            "design_gtech":src_gtech,
            "design_aig":src_aig
        }
        path_raw_pt = os.path.join(self.processed_dir, f"{desgin}_raw_{self.recipe_size}.pt")
        torch.save(data_raw, path_raw_pt)
        
        # load the recipes parallelly
        recipe_list = []
        for logic in self.logics:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for i in range(self.recipe_size):
                    futures.append( executor.submit(self.load_one_logic, folder, logic, desgin, i) )
                logic_list = [future.result() for future in tqdm(futures, desc=f"loading {desgin} {logic}")]
                path_logic_pt = os.path.join(self.processed_dir, f"{desgin}_{logic}_{self.recipe_size}.pt")
                torch.save(logic_list, path_logic_pt)
                recipe_list.append(logic_list)
        
        data = {
            "design_name": desgin,
            "design_gtech": src_gtech,
            "design_aig": src_aig,
            "design_recipes": recipe_list
        }
        self.data_list.append( data )

    def load_one_logic(self, folder, logic, design, index):
        path_one_design = os.path.join(folder, design)
        isOK = True
        pack = {}
        
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
            print("design recipe not complete: ", f"{design}_recipe_{index}")
            isOK = False

        if isOK:
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
    logics = ["abc", "aig", "oig", "xag", "primary", "mig", "gtg"]
    design_list = [
        "ctrl",
        "steppermotordrive",
        "router",
        "int2float",
        "ss_pcm",
        "usb_phy",
        "sasc",
        "cavlc",
        "simple_spi",
        "priority",
        "adder",
        "i2c",
        "systemcdes",
        "max",
        "bar",
        "spi",
        "fir",
        "wb_dma",
        "des3_area",
        "sin",
        "iir",
        "tv80",
        "ac97_ctrl",
        "arbiter",
        "systemcaes",
        "voter",
        "usb_funct",
        "sha256",
        "mem_ctrl",
        "dynamic_node",
        "square",
        "sqrt",
        "multiplier",
        "aes",
        "fpu",
        "log2",
        "aes_secworks",
        "aes_xcrypt",
        "wb_conmax",
        "tinyRocket",
        "div",
        "ethernet",
        "bp_be",
        "picosoc",
        "vga_lcd",
        "jpeg"
    ]
    db = OpenLS_Dataset(folder, recipe_size, logics, design_list)