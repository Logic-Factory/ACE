import os
import sys
import argparse
import glob
import numpy
import pandas as pd
import configparser

import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class Params:
    """ params configuration of Checking
    """
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self._root = config['FOLDER']['root']
        self._target = config['FOLDER']['target']
        self._logicfactory = config['TOOL']['logicfactory']
        self._yosys = config['TOOL']['yosys']
        self._abc = config['TOOL']['abc']
        self._techmap_stdlib = config['LIB']['techmap_stdlib']
        self._techmap_genlib = config['LIB']['techmap_genlib']
        self._gtech_genlib = config['LIB']['gtech_genlib']
        self._ieda_config = config['CONFIG']['ieda_config']
        self._length = int(config['RECIPE']['length'])
        self._times = int(config['RECIPE']['times'])
        
        if self._root == "":
            raise ValueError("root is empty")
        if self._target == "":
            raise ValueError("target is empty")
        if self._logicfactory == "":
            raise ValueError("logicfactory is empty")
        if self._abc == "":
            raise ValueError("abc is empty")
        if self._techmap_stdlib == "":
            raise ValueError("techmap stdlib is empty")
        if self._techmap_genlib == "":
            raise ValueError("techmap genlib is empty")
        if self._gtech_genlib == "":
            raise ValueError("gtech genlib is empty")
        if self._ieda_config == "":
            raise ValueError("ieda config is empty")
        if self._length == "":
            raise ValueError("recipe_len is empty")
        if self._times == "":
            raise ValueError("recipe_times is empty")

    def __str__(self):
        return f"root: {self._root}, target: {self._target}, logicfactory: {self._logicfactory}, yosys: {self._yosys}, abc: {self._abc}, \
            techmap_stdlib: {self._techmap_stdlib}, techmap_genlib: {self._techmap_genlib}, gtech_genlib: {self._gtech_genlib}, \
            length: {self._length}, times: {self._times}"

    def folder_root(self):
        return self._root
    
    def folder_target(self):
        return self._target

    def tool_logicfactory(self):
        return self._logicfactory

    def tool_yosys(self):
        return self._yosys

    def tool_abc(self):
        return self._abc
    
    def lib_techmap_stdlib(self):
        return self._techmap_stdlib
    
    def lib_techmap_genlib(self):
        return self._techmap_genlib
    
    def lib_gtech_genlib(self):
        return self._gtech_genlib

    def config_ieda(self):
        return self._ieda_config

    def recipe_length(self):
        return self._length
    
    def recipe_times(self):
        return self._times

class Checking(object):
    """ Checking
    """
    def __init__(self, params:Params):
        self.params = params
        self.raw_gtech_name = "raw.gtech.v"
        
    def run(self):
        designs = glob.glob(os.path.join(self.params.folder_root(), '**/*.aig'), recursive=True)
        count = 1
        lost_and_found = []
        for design in designs:
            basename = os.path.basename(design)
            filename = os.path.splitext(basename)[0]
            
            print("checking at: ", filename, "   [", count, "/", len(designs), "]")
            count += 1
            
            target_folder = os.path.join(self.params.folder_target(), filename)
            lost_and_found.extend( self.checking_and_regen(target_folder) )
            
        if len(lost_and_found) > 0:
            print("Lost and found: \n{0}".format(lost_and_found))
            # write to file
            with open("lost_and_found.txt", "w") as f:
                f.write("\n".join(lost_and_found))
                
        else:
            print("All files are generated!")
        return

    def checking_and_regen(self, folder_one_design):
        """
            checking whether all the files are generated, if not, this step will regen these file
        """
        logics = ["abc", "aig", "oig", "aog", "xag", "xog", "primary", "mig", "xmg", "gtg"]
        
        lost_and_found = []
        
        for logic in logics:
            foot_logic = os.path.join(folder_one_design, logic)
            if not os.path.exists(foot_logic):
                lost_and_found.append(foot_logic)
                return lost_and_found
            for index in range(self.params.recipe_times() ):
                file_logic = os.path.join(foot_logic, "recipe_{0}.logic.v".format(index))
                file_logic_graphml = os.path.join(foot_logic, "recipe_{0}.logic.graphml".format(index))
                file_logic_qor = os.path.join(foot_logic, "recipe_{0}.logic.qor.json".format(index))
                file_fpga = os.path.join(foot_logic, "recipe_{0}.fpga.v".format(index))
                file_fpga_graphml = os.path.join(foot_logic, "recipe_{0}.fpga.graphml".format(index))
                file_fpga_qor = os.path.join(foot_logic, "recipe_{0}.fpga.qor.json".format(index))
                file_asic = os.path.join(foot_logic, "recipe_{0}.asic.v".format(index))
                file_asic_graphml = os.path.join(foot_logic, "recipe_{0}.asic.graphml".format(index))
                file_asic_qor = os.path.join(foot_logic, "recipe_{0}.asic.qor.json".format(index))
                file_timing_qor = os.path.join(foot_logic, "recipe_{0}.asic.timing.qor.json".format(index))
                file_power_qor = os.path.join(foot_logic, "recipe_{0}.asic.power.qor.json".format(index))

                if not os.path.exists(file_logic):
                    lost_and_found.append(file_logic)
                if not os.path.exists(file_logic_graphml):
                    lost_and_found.append(file_logic_graphml)
                if not os.path.exists(file_logic_qor):
                    lost_and_found.append(file_logic_qor)
                if not os.path.exists(file_fpga):
                    lost_and_found.append(file_fpga)
                if not os.path.exists(file_fpga_graphml):
                    lost_and_found.append(file_fpga_graphml)
                if not os.path.exists(file_fpga_qor):
                    lost_and_found.append(file_fpga_qor)
                if not os.path.exists(file_asic):
                    lost_and_found.append(file_asic)
                if not os.path.exists(file_asic_graphml):
                    lost_and_found.append(file_asic_graphml)
                if not os.path.exists(file_asic_qor):
                    lost_and_found.append(file_asic_qor)
                if not os.path.exists(file_timing_qor):
                    lost_and_found.append(file_timing_qor)
                if not os.path.exists(file_power_qor):
                    lost_and_found.append(file_power_qor)
        return lost_and_found
    
if __name__ == '__main__':    
    file = sys.argv[1]
    params = Params(file)    
    synthesis = Checking(params)
    synthesis.run()