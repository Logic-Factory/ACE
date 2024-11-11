import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import json
import numpy as np
from tqdm import tqdm 
import subprocess
import zstandard as zstd

class CEC:
    def __init__(self, folder, recipe, abc, liberty):
        self.folder = folder
        self.recipe = int(recipe)
        self.abc = abc
        self.liberty = liberty
        self.logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]
        self.black_list = ["processed_dir"]

    def run(self):
        designs = os.listdir(self.folder)
        
        count = 1
        for design in designs:
            if design in self.black_list:
                continue
            
            print("process at: ", design, "   [", count, "/", len(designs), "]")
            count += 1
            
            path_design_src = os.path.join(self.folder, design)
            for index in range(self.recipe):
                curr_design = ""
                gold_design = ""
                
                if os.path.exists( os.path.join(path_design_src, "abc", f"recipe_{index}.asic.v") ):
                    gold_design = os.path.join(path_design_src, "abc", f"recipe_{index}.asic.v")
                elif os.path.exists( os.path.join(path_design_src, "abc", f"recipe_{index}.asic.v.zst") ):
                    gold_design = os.path.join(path_design_src, "abc", f"recipe_{index}.asic.v.zst")
                else:
                    print("no abc asic netlist file:", os.path.exists( os.path.join(path_design_src, "abc", f"recipe_{index}.asic.v") ))
                    assert False
                
                if gold_design.endswith(".zst"):
                    res_gold_design = gold_design[:-4]
                    with open(gold_design, 'rb') as f_in:
                        with open(res_gold_design, 'wb') as f_out:
                            dctx = zstd.ZstdDecompressor()
                            dctx.copy_stream(f_in, f_out)
                    gold_design = res_gold_design

                for logic in self.logics:
                    
                    if os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.v") ):
                        curr_design = os.path.join(path_design_src, logic, f"recipe_{index}.asic.v")
                    elif os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.v.zst") ):
                        curr_design = os.path.join(path_design_src, logic, f"recipe_{index}.asic.v.zst")
                    else:
                        print(f"no {logic} asic netlist file:", os.path.exists( os.path.join(path_design_src, logic, f"recipe_{index}.asic.v") ))
                        assert False
                
                    if curr_design.endswith(".zst"):
                        res_curr_design = curr_design[:-4]
                        with open(curr_design, 'rb') as f_in:
                            with open(res_curr_design, 'wb') as f_out:
                                dctx = zstd.ZstdDecompressor()
                                dctx.copy_stream(f_in, f_out)
                        curr_design = res_curr_design

                    script = f"read {self.liberty}; read -m {gold_design}; cec -n {curr_design};"
                    cmd = f"{self.abc} -c \"{script}\""
                    log = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if "Networks are equivalent" not in log.stdout:
                        print("CEC failed:", gold_design, curr_design)
                    else:
                        print("CEC passed:", gold_design, curr_design)

if __name__ == "__main__":
    folder = sys.argv[1]
    recipe = sys.argv[2]
    abc = sys.argv[3]
    liberty = sys.argv[4]
    cec = CEC(folder, recipe, abc, liberty)
    cec.run()