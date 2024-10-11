import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import json
import numpy as np
from tqdm import tqdm 

from src.io.load_qor import QoR, load_qor
from src.utils.plot import plot_2d_dots, plot_multi_2d_dots, plot_3d_dots


class ProfileDataset:
    def __init__(self, folder, target, recipe):
        self.folder = folder
        self.target = target
        os.makedirs(self.target, exist_ok=True)
        self.recipe = int(recipe)
        self.logics = ["abc", "aig", "oig", "xag", "primary", "mig", "gtg"]
        self.black_list = ["processed_dir"]

        print("configuration:")
        print(f"folder: {self.folder}")
        print(f"target: {self.target}")
        print(f"recipe: {self.recipe}")

    def run(self):
        designs = os.listdir(self.folder)
        for design in designs:
            if design in self.black_list:
                continue
            logicss_size = []
            logicss_depth = []
            asicss_size = []
            asicss_depth = []
            asicss_area = []
            asicss_timing = []
            asicss_power = []
            fpgass_area = []
            fpgass_delay = []
            
            path_design_tgt = os.path.join(self.target, design)
            
            for logic in self.logics:
                print(f"Processing {design} {logic}")
                logics_size, logics_depth, asics_size, asics_depth, asics_area, asics_timing, asics_power, fpgas_area, fpgas_delay = self.plot_design_logic(design, logic)
                logicss_size.append(logics_size)
                logicss_depth.append(logics_depth)
                asicss_size.append(asics_size)
                asicss_depth.append(asics_depth)
                asicss_area.append(asics_area)
                asicss_timing.append(asics_timing)
                asicss_power.append(asics_power)
                fpgass_area.append(fpgas_area)
                fpgass_delay.append(fpgas_delay)
            
            plot_multi_2d_dots( x_lists= logicss_size, y_lists= logicss_depth, labels= self.logics, title="Qor Distribution of Logic Graph", x_label="size", y_label="depth", save_path = os.path.join(path_design_tgt, "aana.logic.graph.pdf"))
            plot_multi_2d_dots( x_lists= asicss_size,  y_lists= asicss_depth,  labels= self.logics, title="Qor Distribution of ASIC Graph",   x_label="size", y_label="depth", save_path = os.path.join(path_design_tgt, "aana.asic.graph.pdf"))
            plot_multi_2d_dots( x_lists= asicss_area,  y_lists= asicss_timing, labels= self.logics, title="Qor Distribution of ASIC Netlist", x_label="area", y_label="timing",save_path = os.path.join(path_design_tgt, "aana.asic.circuit.pdf"))
            plot_multi_2d_dots( x_lists= fpgass_area,  y_lists= fpgass_delay,  labels= self.logics, title="Qor Distribution of FPGA Netlist", x_label="area", y_label="timing",save_path = os.path.join(path_design_tgt, "aana.fpga.circuit.pdf"))
            
    def plot_design_logic(self, design, logic):
        # plot the distribution of QoR for each logic by different recipes
        path_design_src = os.path.join(self.folder, design)
        path_design_tgt = os.path.join(self.target, design)
        os.makedirs(path_design_tgt, exist_ok=True)
        
        # store the comparation for each recipe
        logics_size = []
        logics_depth = []
        asics_size = []
        asics_depth = []
        asics_area = []
        asics_timing = []
        asics_power = []
        fpgas_area = []
        fpgas_delay = []
        
        for index in range(self.recipe):
            logic_qor_file = ""
            asic_qor_file = ""
            asic_timing_file = ""
            asic_power_file = ""
            fpga_qor_file = ""
            
            if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json") ):
                logic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json.zst") ):
                logic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json.zst")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json.gz") ):
                logic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json.gz")
            else:
                print("no logic qor file:", os.path.join(path_design_src, logic, f"recipe_{index}.logic.qor.json"))
                assert False
                
            if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json") ):
                asic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json.zst") ):
                asic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json.zst")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json.gz") ):
                asic_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json.gz")
            else:
                print("no area qor file:", os.path.join(path_design_src, logic, f"recipe_{index}.asic.qor.json"))
                assert False
                
            if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json") ):
                asic_timing_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json.zst") ):
                asic_timing_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json.zst")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json.gz") ):
                asic_timing_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json.gz")
            else:
                print("no area timing file:", os.path.join(path_design_src, logic, f"recipe_{index}.asic.timing.qor.json"))
                assert False
                
            if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json") ):
                asic_power_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json.zst") ):
                asic_power_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json.zst")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json.gz") ):
                asic_power_file = os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json.gz")
            else:
                print("no area power file:", os.path.join(path_design_src, logic, f"recipe_{index}.asic.power.qor.json"))
                assert False
                
            if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json") ):
                fpga_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json.zst") ):
                fpga_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json.zst")
            elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json.gz") ):
                fpga_qor_file = os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json.gz")
            else:
                print("no fpga qor file:", os.path.join(path_design_src, logic, f"recipe_{index}.fpga.qor.json"))
                assert False
            
            logic_size = load_qor(logic_qor_file).get_size()
            logic_depth = load_qor(logic_qor_file).get_depth()
            asic_size = load_qor(asic_qor_file).get_size()
            asic_depth = load_qor(asic_qor_file).get_delay()
            asic_area = load_qor(asic_qor_file).get_area()
            asic_timing = load_qor(asic_timing_file).get_timing()
            asic_power = load_qor(asic_power_file).get_power_total()
            fpga_area = load_qor(fpga_qor_file).get_area()
            fpga_delay = load_qor(fpga_qor_file).get_depth()
            
            logics_size.append(logic_size)
            logics_depth.append(logic_depth)
            asics_size.append(asic_size)
            asics_depth.append(asic_depth)
            asics_area.append(asic_area)
            asics_timing.append(asic_timing)
            asics_power.append(asic_power)
            fpgas_area.append(fpga_area)
            fpgas_delay.append(fpga_delay)
        
        assert len(logics_size) == len(logics_depth) == len(asics_size) == len(asics_depth) == len(asics_area) == len(asics_timing) == len(asics_power) == len(fpgas_area) == len(fpgas_delay)
        
        plot_2d_dots(x_list=logics_size, y_list=logics_depth, title= f"QoR Distribution of Logic Network ({logic})", x_label="circuit size", y_label="circuit depth", save_path=os.path.join(path_design_tgt, logic+f"_logic.graph.pdf") )
        plot_2d_dots(x_list=asics_size, y_list=asics_depth, title= f"QoR Distribution of ASIC netlist ({logic})", x_label="circuit size", y_label="circuit depth", save_path=os.path.join(path_design_tgt, logic+f"_asic.graph.pdf") )
        plot_2d_dots(x_list=asics_area, y_list=asics_timing, title= f"QoR Distribution of ASIC netlist ({logic})", x_label="circuit area", y_label="circuit delay", save_path=os.path.join(path_design_tgt, logic+f"_asic.circuit.2d.pdf") )
        plot_2d_dots(x_list=fpgas_area, y_list=fpgas_delay, title= f"QoR Distribution of FPGA netlist ({logic})", x_label="circuit area", y_label="circuit delay", save_path=os.path.join(path_design_tgt, logic+f"_fpga.circuit.pdf") )
        
        return logics_size, logics_depth, asics_size, asics_depth, asics_area, asics_timing, asics_power, fpgas_area, fpgas_delay

if __name__ == '__main__':
    folder = sys.argv[1]
    target = sys.argv[2]
    recipe = sys.argv[3]
    profile = ProfileDataset(folder, target, recipe)
    profile.run()
