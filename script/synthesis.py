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

OptCmds_Aig = [
    "refactor",
    "refactor -l",
    "refactor -z",
    "refactor -l -z",
    "rewrite",
    "rewrite -l",
    "rewrite -z",
    "rewrite -l -z",
    "resub",
    "resub -l",
    "resub -z",
    "resub -l -z",
    "balance",
    "balance",
    "balance",
    "balance",
]

OptCmds_Xag = [
    "refactor",
    "refactor -z",
    "refactor",
    "refactor -z",
    "rewrite",
    "rewrite -l",
    "rewrite -z",
    "rewrite -l -z",
    "balance",
    "balance",
    "balance -m",
    "balance -f"
]

OptCmds_Mig = [
    "refactor",
    "refactor -z",
    "refactor",
    "refactor -z",
    "rewrite",
    "rewrite -l",
    "rewrite -z",
    "rewrite -l -z",
]

OptCmds_Xmg = [
    "refactor",
    "refactor -z",
    "refactor",
    "refactor -z",
    "rewrite",
    "rewrite -l",
    "rewrite -z",
    "rewrite -l -z",
    "resub",
    "resub -l",
    "resub -z",
    "resub -l -z",
]
        
def gen_gaussian_numbers(rang:int, size:int):
    if size < 5:
        print("warning: sequence size is too small")
    mean = size / 2
    std_dev = max(size / 4, 1) 
    
    real_size = int(abs(numpy.random.normal(mean, std_dev)))   # the size is follow gaussian distribution
    real_size = real_size % size
    
    numbers = []
    for _ in range(real_size):
        number = numpy.random.randint(0, rang)
        numbers.append(number)
    return numbers

def gen_random_numbers(rang:int, size:int):
    if size < 5:
        print("warning: sequence size is too small")
    numbers = numpy.random.randint(0, rang, size)
    return numbers

class Params:
    """ params configuration of Synthesis
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

class Synthesis(object):
    """ Synthesis
    """
    def __init__(self, params:Params):
        self.params = params
        
    def run(self):
        designs = glob.glob( os.path.join(self.params.folder_root(), '*.aig') )
        for design in designs:
            basename = os.path.basename(design)
            filename = os.path.splitext(basename)[0]
            
            print("process at: ", filename)
            
            target_folder = os.path.join(self.params.folder_target(), filename)
            os.makedirs(target_folder, exist_ok=True)
            self.recipe_one_design(design, filename, target_folder)
        return

    def recipe_one_design(self, design, filename, target_folder):
        """ synthesis the data of one design
            step1: generate the gtech representation of the given design
            step2: boolean representation
            step3: logic optimization
            step4: technology mapping
            step5: physics design
        """
        file_gtech = self.apply_gtech_tans(design, filename, target_folder)
        self.apply_physics_synthesis(file_gtech, target_folder)

    
    def apply_gtech_tans(self, desgin, filename, target_folder):
        """ translate the design to gtech format
        """
        gtech = os.path.join(target_folder, 'raw.gtech.v')
        script = "start; anchor -set yosys; read_aiger -file {0}; hierarchy -auto-top; \
                  rename -top {1}; techmap; abc -exe {2} -genlib {3}; \
                  write_verilog {4}; stop;".format(desgin, 
                                                   filename,
                                                   self.params.tool_abc(),
                                                   self.params.lib_gtech_genlib(),
                                                   gtech)
        cmd = "{0} -c \"{1}\"".format(self.params.tool_logicfactory(), 
                                      script)
        log = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return gtech
    
    def apply_physics_synthesis(self, design_in, target_folder):
        """ physics synthesis
            1. Boolean representation
            2. logic optimization
            3. technology mapping
            4. physics design
        """
        def gen_opt_sequence(cmds:list):
            numbers = gen_gaussian_numbers(len(cmds), self.params.recipe_length())
            sequence = ""
            for number in numbers:
                sequence += cmds[number] + ";"
            return sequence
        
        types_ckt = ["aig", "xag", "mig", "xmg"]
        for ckt in types_ckt:
            print(ckt)
            folder_curr = os.path.join(target_folder, ckt)
            os.makedirs(folder_curr, exist_ok=True)
            
            for i in range(self.params.recipe_times()):
                print("recipe {0}".format(i))
                file_script = os.path.join(folder_curr, "recipe_{0}.script".format(i))
                file_logic = os.path.join(folder_curr, "recipe_{0}.logic.graphml".format(i))
                file_netlist = os.path.join(folder_curr, "recipe_{0}.netlist.graphml".format(i))
                file_physics = os.path.join(folder_curr, "recipe_{0}.physics.graphml".format(i))
                file_gdsii = os.path.join(folder_curr, "recipe_{0}.gdsii".format(i))
                
                script = "start; anchor -set lsils; ntktype -tool lsils -type logic -ntk gtg; read_gtech -file {0};".format(design_in)
                
                # logic optimization
                if ckt == "aig":
                    script += "anchor -set abc; ntktype -tool abc -type strash -ntk aig; strash;"
                    opt_sequence = gen_opt_sequence(OptCmds_Aig)
                    script += opt_sequence
                elif ckt == "xag":
                    script += "anchor -set lsils; ntktype -tool lsils -type strash -ntk xag; strash;"
                    opt_sequence = gen_opt_sequence(OptCmds_Xag)
                    script += opt_sequence
                elif ckt == "mig":
                    script += "anchor -set lsils; ntktype -tool lsils -type strash -ntk mig; strash;"
                    opt_sequence = gen_opt_sequence(OptCmds_Mig)
                    script += opt_sequence
                elif ckt == "xmg":
                    script += "anchor -set lsils; ntktype -tool lsils -type strash -ntk xmg; strash;"
                    opt_sequence = gen_opt_sequence(OptCmds_Xmg)
                    script += opt_sequence
                
                # write the logic network
                script += "write_graphml -file {0};".format(file_logic)
                
                # technology mapping
                if ckt == "aig":
                    script += "anchor -set abc; read_genlib {0}; map_asic; write_graphml -file {1};".format(self.params.lib_techmap_genlib(), file_netlist)
                else:
                    script += "anchor -set lsils; read_genlib {0}; map_asic; write_graphml -file {1};".format(self.params.lib_techmap_genlib(), file_netlist)
                
                # # physics design
                # # TODO: store the physics information
                # script += "anchor -set ieda; logic2netlist; config -file {0}; \
                #           init; sta; floorplan; placement; cts; routing; write_gdsii -file {1}; stop;".format(self.params.config_ieda(), file_gdsii)

                cmd = "{0} -c \"{1}\"".format(self.params.tool_logicfactory(),
                                              script)
                # store the script
                with open(file_script, "w") as f:
                    f.write(cmd)
                log = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return

if __name__ == '__main__':    
    file = sys.argv[1]
    params = Params(file)    
    synthesis = Synthesis(params)
    synthesis.run()