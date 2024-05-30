import os
import re
import random
import argparse
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from scipy.stats import gaussian_kde, truncnorm

################################################################################################################################################################
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf',  
            '#7f7f7f', '#e377c2', '#8c564b', '#bcbd22']
ecolors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#9edae5', 
            '#c7c7c7', '#f7b6d2', '#c49c94', '#dbdb8d']
tex_fonts = {
    "font.family": "serif",
    'font.variant': "small-caps",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "lines.linewidth":2.,
    "lines.markersize":4,
    
    "axes.grid": True, 
    "grid.color": ".9", 
    "grid.linestyle": "--",
    "axes.linewidth":1.5, 
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'savefig.facecolor': 'white'
}        

plt.rcParams.update(tex_fonts)
plt.rcParams['figure.dpi'] = 600

# style={ 0:'-', 1: '-', 2:'-.', 3:'--',  4:':', 5:'-.', 6:'--', 7: ':'}
style={ 0:'--', 1: '-', 3:':', 4:'dashdot',  2:'-', -1:'-'}
color={ 0:'C2', 1: 'C3', 3: 'C0', 4:'C1', 5:'C4', 2:'C5'}

linestyle_tuple = ['-', ':', '-.', '--', '-.','-']
markers = ["*", "+", ".",  "x", "d", "p", "|"]
################################################################################################################################################################


# refactor*4, rewrite*4, resub*4, balance*4, make them all the same weight
OptDict = {
    0: "refactor" ,
    1: "refactor -z" ,
    2: "refactor -l" ,
    3: "refactor -l -z" ,
    4: "rewrite" ,
    5: "rewrite -z" ,
    6: "rewrite -l" ,
    7: "rewrite -l -z" ,
    8: "resub" ,
    9: "resub -z",
    10: "resub -l" ,
    11: "resub -l -z" ,
    12: "balance", 
    13: "balance", 
    14: "balance", 
    15: "balance"
}

class SynthesisRecipeData:
    """_summary_
    """
    def __init__(self, root_dir, res_dir, tool_abc, tool_circuit_pool, tool_circuit2graphml, liberty, len_recipe_one, len_recipe_all):
        self.abs_root_dir = os.path.abspath(root_dir.strip()) 
        self.abs_res_dir = os.path.abspath(res_dir.strip())
        self.abs_tool_abc = os.path.abspath(tool_abc.strip())
        self.abs_tool_circuit_pool = os.path.abspath(tool_circuit_pool.strip())
        self.abs_tool_circuit2graphml = os.path.abspath(tool_circuit2graphml.strip())
        self.len_recipe_one = len_recipe_one
        self.len_recipe_all = len_recipe_all
        self.liberty = liberty

    def run(self):
        """ Synthesis Recipe Dataset Main Flow
        """
        # step 1 , fltering the source AIGs
        aigs = self.extract_aigs()
        
        # step 2: synthesis recipe of the AIG designs parallely
        with ThreadPoolExecutor(max_workers = 16) as executor:
            futures = []
            total_tasks = len(aigs)
            for aig in aigs:
                basename = os.path.basename(aig)
                filename = os.path.splitext(basename)[0]
                res_folder = os.path.join(self.abs_res_dir, filename)
                os.makedirs(res_folder, exist_ok=True)  # make dir of current design to store the synthesis data

                # add this taks into future for parallen processing
                future = executor.submit(self.synthesis_recipe_one_design, aig, res_folder)
                futures.append(future)

            # until all the task end!
            for future in futures:
                future.result()
    
    def extract_aigs(self):
        """Extract the all the required AIGs

        Returns:
            aigs_required: the vector of the required AIG pathes
        """
        
        aigs_all = glob.glob(os.path.join(self.abs_root_dir, '*.aig'))
        aigs_required = []
        for aig in aigs_all:
            if self.check_constraints(aig):
                aigs_required.append(aig)
        return aigs_required

    def check_constraints(self, aig):
        """Checking whether current aig is satisfied with the constraints

        Args:
            aig (str): the path of the current AIG

        Returns:
            Boolean: True or False
        """
        
        # add constraints of the name
        basename = os.path.basename(aig)
        filename = os.path.splitext(basename)[0]
        
        # add constraints of the paramaters
        script = "read_aiger {0}; strash; print_stats;".format(aig)
        command = "{0} -c \"{1}\"".format(self.abs_tool_abc, script)        
        result = subprocess.run(command, shell=True, capture_output=True, text= True)
        output = result.stdout
        and_count = 0
        
        match = re.search(r'and =\s*(\d+)\s*lev =\s*(\d+)', output)
        if match:
            and_count = int(match.group(1))
        else:
            assert False
        
        if and_count >= 1 and and_count <= 5000:   # constraints here
            return True
        else:
            return False

    def gen_random_opt_seq_by_gaussian(self):
        """Generate the random optimization sequence, and the length is constrained by the Gaussian distribution

        Returns:
            scripts: optimization sequence (str)
        """
        
        scripts = ""
        len_opt_dict = len(OptDict)
        
        # generate the len_opt_seq_real size follow the Gaussian distribution
        len_opt_seq_real = 0
        mean = self.len_recipe_one / 2
        std_dev = max(self.len_recipe_one / 4, 1) 
        lower_bound = 1
        upper_bound = self.len_recipe_one
        a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
        while True:
            try:
                len_opt_seq_real = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
                len_opt_seq_real = int(round(len_opt_seq_real))
            except Exception as e:
                print("generate normal data wrong!")
                len_opt_seq_real = mean
            if 0 < len_opt_seq_real <= self.len_recipe_one:
                break
        
        # generage the random optimization sequence
        for i in range(len_opt_seq_real):
            rnumber = rnumber = random.randint(0, len_opt_dict - 1)
            scripts += OptDict[rnumber] + ';'
        return scripts, len_opt_seq_real
    
    def gen_random_opt_seq_by_fixed_size(self):
        """Generate the random optimization sequence, and the length is the same with the given size
        
        Returns:
            scripts: optimization sequence (str)
        """
        
        scripts = ""
        len_opt_dict = len(OptDict)
        for i in range(self.len_recipe_one):
            rnumber = rnumber = random.randint(0, len_opt_dict)
            scripts += OptDict[rnumber] + ';'
        return scripts
    
    def apply_abc_optimization(self, aig_in, opt_script, aig_out):
        """ Apply the logic optimization of one given AIG and the optimization sequence
        
        Args:
            aig_in (str): path of the source AIG file
            aig_out (str): path of the output AIG file
            opt_script (str): the optimization scripts
        Returns:
            and_count, lev_count
        """
        
        script = "read_aiger {0}; strash; {1}; print_stats; write_aiger {2}".format(aig_in, opt_script, aig_out)
        command = "{0} -c \"{1}\"".format(self.abs_tool_abc, script)
        
        result = subprocess.run(command, shell=True, capture_output=True, text= True)

        output = result.stdout
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_output = ansi_escape.sub('', output)
        
        pattern_opt_stats = r'and =\s*(\d+)\s*lev =\s*(\d+)'
        match = re.search(pattern_opt_stats, clean_output)
        assert match
        
        opt_gate_count = int(match.group(1))
        opt_lev_count  = int(match.group(2))
        return opt_gate_count, opt_lev_count

    def apply_abc_asic_mapping(self, aig_in, netlist_out):
        """Apply the ASIC mapping of the given AIG

        Args:
            aig_in (str): path of the input AIG file 
            netlist_out (str): path of the output Gate-level netlist file
        """
        # read_lib ../src/liberty/asap7_clean.lib
        # read_aiger ../benchmark/kdd24/adder.aig
        # strash
        # map
        # print_stats # ../benchmark/kdd24/adder      : i/o =  256/  129  lat =    0  nd =   585  edge =   1231  area =898.31  delay =2613.78  lev = 129
        # stime       # WireLoad = "none"  Gates =    585 ( 11.8 %)   Cap =  0.6 ff (  2.1 %)   Area =      898.13 ( 88.2 %)   Delay =  3770.65 ps  ( 23.9 %)
        # write_verilog test.v
        
        script = "read_lib {0}; read_aiger {1}; strash; map; print_stats; stime; write_verilog {2}".format(self.liberty, aig_in, netlist_out)
        command = "{0} -c \"{1}\"".format(self.abs_tool_abc, script)
        
        result = subprocess.run(command, shell=True, capture_output=True, text= True)
        output = result.stdout
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_output = ansi_escape.sub('', output)
        
        pattern_map_stats = r"i/o\s*=\s*(\d+)/\s*(\d+)\s*lat\s*=\s*(\d+)\s*nd\s*=\s*(\d+)\s*edge\s*=\s*(\d+)\s*area\s*=\s*([\d.]+)\s*delay\s*=\s*([\d.]+)\s*lev\s*=\s*(\d+)"
        pattern_stime_stats = r"Gates\s*=\s*(\d+)\s*\(\s*([\d.]+)\s*%\)\s*Cap\s*=\s*([\d.]+)\s*ff\s*\(\s*([\d.]+)\s*%\)\s*Area\s*=\s*([\d.]+)\s*\(\s*([\d.]+)\s*%\)\s*Delay\s*=\s*([\d.]+)\s*ps\s*\(\s*([\d.]+)\s*%\)"

        match_map = re.search(pattern_map_stats, clean_output)
        match_stime = re.search(pattern_stime_stats, clean_output)
        assert match_map
        assert match_stime
    
        map_gate_count = int(match_map.group(4))
        map_edge_count = int(match_map.group(5))
        map_area_count = float(match_map.group(6))
        map_delay_count = float(match_map.group(7))
        map_lev_count = int(match_map.group(8))

        stime_cap_count = float(match_stime.group(3))
        stime_cap_percent = float(match_stime.group(4))
        stime_area_count = float(match_stime.group(5))
        stime_area_percent = float(match_stime.group(6))
        stime_delay_count = float(match_stime.group(7))
        stime_delay_percent = float(match_stime.group(8))
        return [map_gate_count, map_edge_count, map_area_count, map_delay_count, map_lev_count], [stime_cap_count, stime_cap_percent, stime_area_count, stime_area_percent, stime_delay_count, stime_delay_percent]

    def apply_abc_fpga_mapping(self, aig_in, netlist_out):
        """Apply the FPGA mapping of the given AIG

        Args:
            aig_in (str): path of the input AIG file 
            netlist_out (str): path of the output Gate-level netlist file
        """
        # read_aiger ../benchmark/kdd24/adder.aig
        # strash
        #  if -K 6
        # print_stats # ../benchmark/kdd24/adder      : i/o =  256/  129  lat =    0  nd =   254  edge =   1036  aig  =  1666  lev = 51
        # write_blif test.blif
        
        script = "read_aiger {0}; strash; if -K 6; print_stats; write_blif {1}".format(aig_in, netlist_out)
        command = "{0} -c \"{1}\"".format(self.abs_tool_abc, script)
        
        result = subprocess.run(command, shell=True, capture_output=True, text= True)
        output = result.stdout
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_output = ansi_escape.sub('', output)
        
        pattern_if_stats = r"i/o\s*=\s*(\d+)/\s*(\d+)\s*lat\s*=\s*(\d+)\s*nd\s*=\s*(\d+)\s*edge\s*=\s*(\d+)\s*aig\s*=\s*(\d+)\s*lev\s*=\s*(\d+)"

        match_if = re.search(pattern_if_stats, clean_output)
        assert match_if
    
        if_gate_count = int(match_if.group(4))
        if_edge_count = int(match_if.group(5))
        if_aig_count  = int(match_if.group(6))
        if_lev_count  = int(match_if.group(7))
        return [if_gate_count, if_edge_count, if_aig_count, if_lev_count]
        
    def apply_circuit_pooling(self, aig_in, res_pool):
        """ Using circuit pooling to generate the flat graph representation

        Args:
            aig_in (_type_): _description_
            graphml_out (_type_): _description_
        """      
        command_aig_2_graphml = f"{self.abs_tool_circuit2graphml} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_aig.graphml')}"
        command_bdg_dependent_graph = f"{self.abs_tool_circuit_pool} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_bdg.graphml')} {'o'} {'0'}"
        command_bdg_homo_1_dependent_graph = f"{self.abs_tool_circuit_pool} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_bdg_homo_1.graphml')} {'o'} {'1'}"
        command_bdg_homo_2_dependent_graph = f"{self.abs_tool_circuit_pool} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_bdg_homo_2.graphml')} {'o'} {'2'}"
        command_port_dependent_graph = f"{self.abs_tool_circuit_pool} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_port.graphml')} {'o'} {'-1'}"
        command_heter_dependent_graph = f"{self.abs_tool_circuit_pool} {aig_in} {os.path.join(self.abs_res_dir, res_pool + '_heter.graphml')} {'e'}"
        
        commands = {
                'aig' : command_aig_2_graphml,
                'bdg' : command_bdg_dependent_graph,
                'bdg_homo_1' : command_bdg_homo_1_dependent_graph,
                'bdg_homo_2' : command_bdg_homo_2_dependent_graph,
                'port' : command_port_dependent_graph,
                'heter' : command_heter_dependent_graph
                }

        def extract_number(pattern, text):
            match = re.search(pattern, text)
            return match.group(1) if match else None
        
        types = []
        gates_size_list = []
        depths_size_list = []
        constant_size_list = []
        input_size_list = []
        output_size_list = []
                
        for cmd_type, command in commands.items():
            result = subprocess.run(command, shell=True, capture_output=True, text= True)
            output = result.stdout
            ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
            clean_output = ansi_escape.sub('', output)
            
            constant_size = extract_number(r'constant size: (\d+)', clean_output)
            input_size = extract_number(r'input size: (\d+)', clean_output)
            output_size = extract_number(r'output size: (\d+)', clean_output)
            gates_origin = extract_number(r'gates origin: (\d+)', clean_output)
            gates_pooling = extract_number(r'gates pooling: (\d+)', clean_output)
            depth_origin = extract_number(r'depth origin: (\d+)', clean_output)
            depth_pooling = extract_number(r'depth pooling: (\d+)', clean_output)    
            
            constant_size_list.append(constant_size)
            input_size_list.append(input_size)
            output_size_list.append(output_size)
            
            types.append(cmd_type)
            gates_size_list.append(gates_pooling)
            depths_size_list.append(depth_pooling)
        
        # store the statistics of the circuit pooling
        data = {
            "type" : types,
            "constant" : constant_size_list,
            "input" : input_size_list,
            "output" : output_size_list,
            "gates" : gates_size_list,
            "depth" : depths_size_list
        }
        df = pd.DataFrame(data)
        res_csv = os.path.join(res_pool + ".csv")
        df.to_csv(res_csv)
    
    def synthesis_recipe_one_design(self, aig, res_folder):
        """Synthesis recipe one AIG design to generate the following datas:
            1. different structured AIGs of the current design (under the constraints of Boolean equivalence)
            2. asic mapping stats + stime stats
            3. fpga mapping stats
            4. circuit_pooling stats

        Args:
            aig (str): _description_
            res_folder (str): _description_
        """
        recipe_files = []   # store the filename of each recipe, only store the filename, not include the suffix name
        
        opt_lens_cnt = []  # store the length of each optimization script
        opt_gates_cnt = []
        opt_levs_cnt = []
        
        asic_gates_cnt = []
        asic_edges_cnt = []
        asic_areas_cnt = []
        asic_delays_cnt = []
        asic_levels_cnt = []
        
        stime_caps_cnt = []
        stime_caps_pct = []
        stime_areas_cnt = []
        stime_areas_pct = []
        stime_delays_cnt = []
        stime_delays_pct = []

        fpga_gates_cnt = []
        fpga_edges_cnt = []
        fpga_aigs_cnt = []
        fpga_levs_cnt = []

        basename = os.path.basename(aig)
        filename = os.path.splitext(basename)[0]
        
        # for i in range(self.len_recipe_all):
        scriptDict = []
        for i in tqdm(range(self.len_recipe_all), desc=filename):
            recipe_filename = filename + "_recipe_" + str(i)
            opt_aig = os.path.join(res_folder, recipe_filename + ".aig")
            res_asic = os.path.join(res_folder, recipe_filename + "_asic.v")
            res_fpga = os.path.join(res_folder, recipe_filename + "_fpga.blif")
            res_pooling = os.path.join(res_folder, recipe_filename + "_pool")
            
            # write the script (generate the individual opt sequence)
            while True: 
                opt_script, opt_script_len = self.gen_random_opt_seq_by_gaussian()
                if opt_script not in scriptDict:
                    scriptDict.append(opt_script)
                    break
            res_script = os.path.join(res_folder, recipe_filename + ".script")            
            with open(res_script, "w") as f:
                f.write(opt_script)
            
            # synthesis current AIG
            opt_gate_count, opt_lev_count = self.apply_abc_optimization(aig, opt_script, opt_aig)
            asic_stats, stime_stats = self.apply_abc_asic_mapping(opt_aig, res_asic)
            fpga_stats = self.apply_abc_fpga_mapping(opt_aig, res_fpga)
            self.apply_circuit_pooling(opt_aig, res_pooling)

            recipe_files.append(recipe_filename)
            
            opt_lens_cnt.append(opt_script_len)
            opt_gates_cnt.append(opt_gate_count)
            opt_levs_cnt.append(opt_lev_count)
            
            asic_gates_cnt.append(asic_stats[0])
            asic_edges_cnt.append(asic_stats[1]) 
            asic_areas_cnt.append(asic_stats[2]) 
            asic_delays_cnt.append(asic_stats[3])
            asic_levels_cnt.append(asic_stats[4])
            
            stime_caps_cnt.append(stime_stats[0])
            stime_caps_pct.append(stime_stats[1])
            stime_areas_cnt.append(stime_stats[2]) 
            stime_areas_pct.append(stime_stats[3]) 
            stime_delays_cnt.append(stime_stats[4])
            stime_delays_pct.append(stime_stats[5])
            
            fpga_gates_cnt.append(fpga_stats[0])
            fpga_edges_cnt.append(fpga_stats[1])
            fpga_aigs_cnt.append(fpga_stats[2]) 
            fpga_levs_cnt.append(fpga_stats[3])
            
        # make the stats_folder to store the statistics data
        stats_folder = os.path.join(res_folder, "stats")
        os.makedirs(stats_folder, exist_ok=True) 
        
        # store the statistics of the synthesis recipes of current design to a csv file
        data = {
            "file" : recipe_files,
            "opt_lens_cnt" : opt_lens_cnt,
            "opt_gates_cnt" : opt_gates_cnt,
            "opt_lev_cnt" : opt_levs_cnt,
            "asic_gates_cnt" : asic_gates_cnt,
            "asic_edges_cnt" : asic_edges_cnt,
            "asic_areas_cnt" : asic_areas_cnt,
            "asic_delays_cnt" : asic_delays_cnt,
            "asic_levels_cnt" : asic_levels_cnt,
            "stime_caps_cnt" : stime_caps_cnt,
            "stime_caps_pct" : stime_caps_pct,
            "stime_areas_cnt" : stime_areas_cnt ,
            "stime_areas_pct" : stime_areas_pct ,
            "stime_delays_cnt" : stime_delays_cnt,
            "stime_delays_pct" : stime_delays_pct,
            "fpga_gates_cnt" : fpga_gates_cnt,
            "fpga_edges_cnt" : fpga_edges_cnt,
            "fpga_aigs_cnt" : fpga_aigs_cnt,
            "fpga_levs_cnt" : fpga_levs_cnt
        }
        df = pd.DataFrame(data)
        res_csv = os.path.join(stats_folder, "recipes.csv")
        df.to_csv(res_csv)
    
        # draw the figures of the data
        
        ## data1.1: Optimization stats Distribution
        path_opt_stats_distr = os.path.join(stats_folder, "opt_stats_distribution.pdf")
        self.draw_2d_distribution(data_x= opt_gates_cnt, label_x= "Gate", data_y=opt_levs_cnt, label_y="Depth", path=path_opt_stats_distr)
        
        ## data1.2: Optimization length Distribution
        path_opt_len_distr = os.path.join(stats_folder, "opt_len_distribution.pdf")
        self.draw_1d_distribution(data_x= opt_lens_cnt, label_x = "len of opt sequence", path = path_opt_len_distr)

        ## data2.1: ASIC Mapping stats Distribution        
        path_asic_map_stats_distr = os.path.join(stats_folder, "asic_map_stats_distribution.pdf")
        self.draw_2d_distribution(data_x= asic_areas_cnt, label_x= "Area", data_y=asic_delays_cnt, label_y="Delay", path=path_asic_map_stats_distr)
        
        ## data2.2: Stime stats Distribution
        path_asic_stime_stats_distr = os.path.join(stats_folder, "asic_stime_stats_distribution.pdf")
        self.draw_2d_distribution(data_x= stime_areas_cnt, label_x= "Area", data_y=stime_delays_cnt, label_y="Delay", path=path_asic_stime_stats_distr)
                
        ## data3: FPGA Mapping stats Distribution
        path_fpga_map_stats_distr = os.path.join(stats_folder, "fpga_map_stats_distribution.pdf")
        self.draw_2d_distribution(data_x= fpga_gates_cnt, label_x= "Area", data_y=fpga_levs_cnt, label_y="Delay", path=path_fpga_map_stats_distr)
        
        ## data4: Circuit Pooling stats Distribution
        path_circuit_pooling_stats_distr = os.path.join(stats_folder, "circuit_pool_stats_distribution.pdf")
    
    def draw_2d_distribution(self, data_x, label_x, data_y, label_y, path):
        data = np.vstack([data_x, data_y])
        try:
            kde = gaussian_kde(data)
            z = kde(data)
            fig, ax = plt.subplots()
            sc = ax.scatter(data_x, data_y, c=z, cmap='viridis', alpha=0.7)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            fig.colorbar(sc, label='Density')
            fig.savefig(path)
            plt.close(fig)
        except Exception as e:
            points = list(zip(data_x, data_y))
            counts = Counter(points)
            unique_points, counts = zip(*counts.items())
            unique_ands, unique_levs = zip(*unique_points)
            colors = np.array(counts)
            fig, ax = plt.subplots()
            scatter = ax.scatter(unique_ands, unique_levs, c=colors, cmap='viridis', alpha=0.7)
            ax.scatter(data_x, data_y, alpha=0.7)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            fig.colorbar(scatter, label='Count')
            fig.savefig(path)
            plt.close(fig)

    def draw_1d_distribution(self, data_x, label_x, path):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data_x, bins=range( 0, self.len_recipe_one ), edgecolor='black', alpha=0.7)
        ax.set_xlabel(label_x)
        ax.set_ylabel('Frequency')
        ax.grid(False)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlim(0, self.len_recipe_one)
        # ax.set_ylim(0, self.len_recipe_one)
        fig.savefig(path)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Open Logic Synthesi Dataset", description="Synthesis Recipes of the AIG designs")
    parser.add_argument('--root_dir', type=str, required=True, help='path of the AIG benchmark directory')
    parser.add_argument('--res_dir', type=str, required=True, help='path of the result dataset directory')
    parser.add_argument('--tool_abc', type=str, required=True, help='path of the berkeley-abc tool')
    parser.add_argument('--tool_circuit_pool', type=str, required=True, help='path of the circuit_pooling tool')
    parser.add_argument('--tool_circuit2graphml', type=str, required=True, help='path of the circuit2graphml tool')
    parser.add_argument('--liberty', type=str, required=True,help='path of the liberty file')
    parser.add_argument('--len_recipe_one', type=int, required=False, default=20, help='length of the max optimization sequence for one synthesis recipe, default = 20')
    parser.add_argument('--len_recipe_all', type=int, required=False, default=500, help='length of the synthesis recipe times of one AIG design, default = 500')
    args = parser.parse_args()
    
    synthesis_data = SynthesisRecipeData(args.root_dir, args.res_dir, args.tool_abc, args.tool_circuit_pool, args.tool_circuit2graphml, args.liberty, args.len_recipe_one, args.len_recipe_all)
    synthesis_data.run()