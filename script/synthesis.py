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
        designs = glob.glob(os.path.join(self.params.folder_root(), '**/*.aig'), recursive=True)
        count = 1
        for design in designs:
            basename = os.path.basename(design)
            filename = os.path.splitext(basename)[0]
            
            print("process at: ", filename, "   [", count, "/", len(designs), "]")
            count += 1
            
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
        # get the raw gtech first
        file_gtech = self.apply_gtech_tans(design, filename, target_folder)
        # synthesis the sequence and internal designs
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
        logics_root = ["abc"]
        logics_aux = ["aig", "oig", "aog", "xag", "xog", "primary", "mig", "xmg", "gtg"]
        aigs_synthesised = []
        for logic_root in logics_root:
            folder_aig = os.path.join(target_folder, logic_root)
            os.makedirs(folder_aig, exist_ok=True)
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range( self.params.recipe_times()):
                    # make sure the logic_aux is corresponding to root aig
                    file_logic_aig = os.path.join(folder_aig, "recipe_{0}.logic.aig".format(i))
                    aigs_synthesised.append(file_logic_aig)
                    future = executor.submit(self.process_logic_root, design_in, i, folder_aig)
                    futures.append(future)
                
                for future in tqdm( futures, desc= "abc ing"):
                    future.result()
                
        # Boolean representation
        for logic_aux in logics_aux:
    
            folder_logic = os.path.join(target_folder, logic_aux)
            os.makedirs(folder_logic, exist_ok=True)
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range( len(aigs_synthesised) ):
                    future = executor.submit(self.process_logic_aux, aigs_synthesised, i, logic_aux, folder_logic)
                    futures.append(future)

                for future in tqdm( futures, desc= logic_aux + " ing"):
                    future.result()
        return

    def process_logic_root(self, design_in, index, folder_root):
        
        def gen_opt_sequence(cmds:list):
            numbers = gen_gaussian_numbers(len(cmds), self.params.recipe_length())
            sequence = ""
            for number in numbers:
                sequence += cmds[number] + ";"
            return sequence
        
        file_script = os.path.join(folder_root, "recipe_{0}.script".format(index))
        file_logic = os.path.join(folder_root, "recipe_{0}.logic.v".format(index))
        file_logic_aig = os.path.join(folder_root, "recipe_{0}.logic.aig".format(index))
        file_logic_graphml = os.path.join(folder_root, "recipe_{0}.logic.graphml".format(index))
        file_logic_dot = os.path.join(folder_root, "recipe_{0}.logic.dot".format(index))
        file_logic_qor = os.path.join(folder_root, "recipe_{0}.logic.qor.json".format(index))
        file_fpga = os.path.join(folder_root, "recipe_{0}.fpga.v".format(index))
        file_fpga_graphml = os.path.join(folder_root, "recipe_{0}.fpga.graphml".format(index))
        file_fpga_dot = os.path.join(folder_root, "recipe_{0}.fpga.dot".format(index))
        file_fpga_qor = os.path.join(folder_root, "recipe_{0}.fpga.qor.json".format(index))
        file_asic = os.path.join(folder_root, "recipe_{0}.asic.v".format(index))
        file_asic_graphml = os.path.join(folder_root, "recipe_{0}.asic.graphml".format(index))
        file_asic_dot = os.path.join(folder_root, "recipe_{0}.asic.dot".format(index))
        file_asic_qor = os.path.join(folder_root, "recipe_{0}.asic.qor.json".format(index))
        # physics QoR
        file_physics_qor = os.path.join(folder_root, "recipe_{0}.physics.qor.json".format(index))    # asic timing / area
        
        script = "start; anchor -tool lsils; ntktype -tool lsils -stat logic -type gtg; read_gtech -file {0}; ".format(design_in)
        
        # logic optimization
        script += "anchor -tool abc; ntktype -tool abc -stat strash -type aig; update -n; strash; "
        script += gen_opt_sequence(OptCmds_Aig)
        
        # write the logic network
        script += " write_dot -file {0}; write_graphml -file {1}; write_aiger -file {2}; write_verilog -file {3}; print_stats -file {4};".format( file_logic_dot, file_logic_graphml, file_logic_aig, file_logic, file_logic_qor)
        
        # set the anchor
        script += "anchor -tool abc; "
        
        # technology mapping
        script +=  "ntktype -tool abc -stat strash -type aig; strash; map_fpga; ntktype -tool abc -stat netlist -type fpga; write_dot -file {0}; write_graphml -file {1}; write_verilog -K 6 -f -file {2};  print_stats -file {3};".format( file_fpga_dot, file_fpga_graphml, file_fpga, file_fpga_qor)
        script +=  "ntktype -tool abc -stat strash -type aig; strash; read_genlib {0}; map_asic; ntktype -tool abc -stat netlist -type asic; write_dot -file {1}; write_graphml -file {2}; write_verilog -file {3};  print_stats -file {4}; ".format(self.params.lib_techmap_genlib(), file_asic_dot, file_asic_graphml, file_asic, file_asic_qor)
        
        # physics design
        # TODO: store the physics information
        script += "anchor -set ieda; logic2netlist; config -file {0}; init; sta; power; print_stats -file {1};".format(self.params.config_ieda(), file_physics_qor)
        script += "stop;"
        
        cmd = "{0} -c \"{1}\"".format(self.params.tool_logicfactory(),
                                    script)
        # store the script
        with open(file_script, "w") as f:
            f.write(cmd)
        log = subprocess.run(cmd, shell=True, capture_output=True, text=True)



    def process_logic_aux(self, aigs, index, logic_aux, folder_aux):
        aig_curr = aigs[index]
        
        file_script = os.path.join(folder_aux, "recipe_{0}.script".format(index))
        file_logic = os.path.join(folder_aux, "recipe_{0}.logic.v".format(index))
        file_logic_graphml = os.path.join(folder_aux, "recipe_{0}.logic.graphml".format(index))
        file_logic_dot = os.path.join(folder_aux, "recipe_{0}.logic.dot".format(index))
        file_logic_qor = os.path.join(folder_aux, "recipe_{0}.logic.qor.json".format(index))
        file_fpga = os.path.join(folder_aux, "recipe_{0}.fpga.v".format(index))
        file_fpga_graphml = os.path.join(folder_aux, "recipe_{0}.fpga.graphml".format(index))
        file_fpga_dot = os.path.join(folder_aux, "recipe_{0}.fpga.dot".format(index))
        file_fpga_qor = os.path.join(folder_aux, "recipe_{0}.fpga.qor.json".format(index))
        file_asic = os.path.join(folder_aux, "recipe_{0}.asic.v".format(index))
        file_asic_graphml = os.path.join(folder_aux, "recipe_{0}.asic.graphml".format(index))
        file_asic_dot = os.path.join(folder_aux, "recipe_{0}.asic.dot".format(index))
        file_asic_qor = os.path.join(folder_aux, "recipe_{0}.asic.qor.json".format(index))
        file_physics_qor = os.path.join(folder_aux, "recipe_{0}.physics.qor.json".format(index))    # asic timing / area
        
        script = "start; anchor -tool lsils; ntktype -tool lsils -stat logic -type aig; read_aiger -file {0}; ".format(aig_curr)
        if logic_aux == "mig" or logic_aux == "xmg":
            script += "convert -from {0} -to {1} -n; ".format("aig", logic_aux)
        else:
            script += "convert -from {0} -to {1}; ".format("aig", logic_aux)
        script += "ntktype -tool lsils -stat logic -type {0}; ".format(logic_aux)
                        
        # write the logic network
        script += "write_dot -file {0}; write_graphml -file {1}; write_verilog -file {2}; print_stats -file {3};".format(file_logic_dot, file_logic_graphml, file_logic, file_logic_qor)
        
        # set the anchor
        script += "anchor -tool lsils; strash; "
        
        # technology mapping
        script +=  "ntktype -tool lsils -stat strash -type {0}; strash; map_fpga; ntktype -tool lsils -stat netlist -type fpga; write_dot -file {1}; write_graphml -file {2}; write_verilog -file {3}; print_stats -file {4};".format(logic_aux, file_fpga_dot, file_fpga_graphml, file_fpga, file_fpga_qor )
        script +=  "ntktype -tool lsils -stat strash -type {0}; strash; read_genlib {1}; map_asic; ntktype -tool lsils -stat netlist -type asic; write_dot -file {2}; write_graphml -file {3}; write_verilog -file {4}; print_stats -file {5};".format(logic_aux, self.params.lib_techmap_genlib(), file_asic_dot, file_asic_graphml, file_asic, file_asic_qor)
        
        # physics design
        # TODO: store the physics information
        script += "anchor -set ieda; logic2netlist; config -file {0}; init; sta; power; print_stats -file {1};".format(self.params.config_ieda(), file_physics_qor)

        script += "stop;"
        
        cmd = "{0} -c \"{1}\"".format(self.params.tool_logicfactory(),
                                      script)
        # store the script
        with open(file_script, "w") as f:
            f.write(cmd)
        log = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if __name__ == '__main__':    
    file = sys.argv[1]
    params = Params(file)    
    synthesis = Synthesis(params)
    synthesis.run()