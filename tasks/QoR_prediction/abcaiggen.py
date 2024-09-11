import os
import re
import random
import glob
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde, truncnorm
from concurrent.futures import ThreadPoolExecutor, as_completed

# refactor*4, rewrite*4, resub*4, balance*4, make them all the same weight
OptDict = {
    0: "refactor",
    1: "refactor -z",
    2: "refactor -l",
    3: "refactor -l -z",
    4: "rewrite",
    5: "rewrite -z",
    6: "rewrite -l",
    7: "rewrite -l -z" ,
    8: "resub",
    9: "resub -z",
    10: "resub -l",
    11: "resub -l -z",
    12: "balance",
    13: "balance", 
    14: "balance", 
    15: "balance"
}

def normalize_and_append(df, column_name, new_column_name):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[[column_name]])
    df[new_column_name] = normalized_data.ravel()

class SynthesisRecipeData:
    """_summary_
    """
    def __init__(self, source_dir, res_dir, tool_abc, liberty, len_recipe_one, len_recipe_all):
        self.abs_source_dir = os.path.abspath(source_dir.strip()) 
        self.abs_res_dir = os.path.abspath(res_dir.strip())
        self.abs_tool_abc = os.path.abspath(tool_abc.strip())
        self.len_recipe_one = len_recipe_one
        self.len_recipe_all = len_recipe_all
        self.liberty = liberty
    
    def run(self,and_num):
        """ Synthesis Recipe Dataset Main Flow
        """
        # step 1 , fltering the source AIGs
        aigs = self.extract_aigs(and_num)
        # step 2: synthesis recipe of the AIG designs parallely
        with ThreadPoolExecutor(max_workers=32) as executor:
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

        return len(aigs)
    
    def extract_aigs(self, and_num):
        """Extract the all the required AIGs

        Returns:
            aigs_required: the vector of the required AIG pathes
        """
        
        aigs_all = glob.glob(os.path.join(self.abs_source_dir, '*.aig'))
        aigs_required = []
        for aig in aigs_all:
            if self.check_constraints(aig, and_num):
                aigs_required.append(aig)
        return aigs_required
    
    def check_constraints(self, aig, and_num):
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
            # print(aig)
            assert False
        
        if and_count >= 5000 and and_count <= and_num:   # constraints here
        # if and_count >= 1:   # constraints here

            return True
        else:
            return False
    
    def gen_random_opt_seq_by_gaussian(self):
        """Generate the random optimization sequence, and the length is constrained by the Gaussian distribution

        Returns:
            scripts: optimization sequence (str)
        """
        
        scripts = ""
        scripts_num = ""
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
            scripts_num += str(rnumber) + ','
        return scripts, scripts_num, len_opt_seq_real
    
    def gen_random_opt_seq_by_fixed_size(self):
        """Generate the random optimization sequence, and the length is the same with the given size
        
        Returns:
            scripts: optimization sequence (str)
        """
        
        scripts = ""
        scripts_num = ""
        len_opt_dict = len(OptDict) - 1
        for i in range(self.len_recipe_one):
            rnumber = random.randint(0, len_opt_dict)
            scripts += OptDict[rnumber] + ';'
            scripts_num += str(rnumber) + ','
        return scripts, scripts_num

    def apply_abc_optimization(self, aig_in, opt_script, aig_out):
        """ Apply the logic optimization of one given AIG and the optimization sequence
        
        Args:
            aig_in (str): path of the source AIG file
            aig_out (str): path of the output AIG file
            opt_script (str): the optimization scripts
        Returns:
            and_count, lev_count
        """
        
        script = "read_aiger {0}; read_lib {1}; strash; {2} write_aiger {3}; map; print_stats".format(aig_in, self.liberty, opt_script, aig_out)
        command = "{0} -c \"{1}\"".format(self.abs_tool_abc, script)
        result = subprocess.run(command, shell=True, capture_output=True, text= True)
        output = result.stdout

        area_match = re.search(r'area =([\d\.]+)', output)
        delay_match = re.search(r'delay =([\d\.]+)', output)

        area = float(area_match.group(1)) if area_match else None
        delay = float(delay_match.group(1)) if delay_match else None
        return area, delay


    def synthesis_recipe_one_design(self, aig, res_folder):

        recipe_files = []

        opt_scripts = []

        # opt_lens_cnt = []  # store the length of each optimization script
        areas = []
        delays = []

        basename = os.path.basename(aig)
        filename = os.path.splitext(basename)[0]
        
        # for i in range(self.len_recipe_all):
        scriptDict = []
        for i in tqdm(range(self.len_recipe_all), desc=filename):
            recipe_filename = filename + "_recipe_" + str(i)
            aig_out = os.path.join(res_folder, recipe_filename + ".aig")
            while True: 
                opt_script, opt_script_num,_= self.gen_random_opt_seq_by_gaussian()
                if opt_script not in scriptDict:
                    scriptDict.append(opt_script)
                    opt_scripts.append(opt_script_num)
                    break
            res_script = os.path.join(res_folder, recipe_filename + ".script")            
            with open(res_script, "w") as f:
                f.write(opt_script)
            
            area, delay = self.apply_abc_optimization(aig, opt_script, aig_out)

            recipe_files.append(recipe_filename)

            # opt_lens_cnt.append(opt_script_len)
            areas.append(area)
            delays.append(delay)
            
        # make the stats_folder to store the statistics data
        stats_folder = os.path.join(res_folder, "stats")
        os.makedirs(stats_folder, exist_ok=True) 
        

        data = {
            "file" : recipe_files,
            "opt_script" : opt_scripts,
            "areas" : areas,
            "delays" : delays,
        }
        df = pd.DataFrame(data)
        normalize_and_append(df, "areas", "areas_norm")
        normalize_and_append(df, "delays", "delays_norm")
        res_csv = os.path.join(stats_folder, "recipes.csv")
        df.to_csv(res_csv)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Open Logic Synthesi Dataset", description="Synthesis Recipes of the AIG designs")
    parser.add_argument('--source_dir', type=str, default='../../benchmark/comb', help='path of the AIG benchmark directory')
    parser.add_argument('--design_class', type=str, default='comb', choices=['comb','core'], help='class of the AIG benchmark')
    parser.add_argument('--res_dir', type=str, default='../../data/aig', help='path of the result dataset directory')
    parser.add_argument('--tool_abc', type=str, default='', help='path of the berkeley-abc tool')
    parser.add_argument('--liberty', type=str, default='../../techlib/asap7.lib',help='path of the liberty file')
    parser.add_argument('--len_recipe_one', type=int, required=False, default=20, help='length of the max optimization sequence for one synthesis recipe, default = 20')
    parser.add_argument('--len_recipe_all', type=int, required=False, default=500, help='length of the synthesis recipe times of one AIG design, default = 500')
    parser.add_argument('--and_num',type=int, required=False, default = 10000, help='Maximum number of and gates')

    args = parser.parse_args()
    
    res_dir = os.path.join(args.res_dir, args.design_class)

    if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    synthesis_data = SynthesisRecipeData(args.source_dir, res_dir, args.tool_abc, args.liberty, args.len_recipe_one, args.len_recipe_all)

    synthesis_data.run(args.and_num)