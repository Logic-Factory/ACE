import os
import glob
import re
import subprocess
import argparse
import pandas as pd

def apply_abc_profile(aig, tool_abc):
    script = "read_aiger {0}; strash; print_stats;".format(aig)
    
    command = "{0} -c \"{1}\"".format(tool_abc, script)
    result = subprocess.run(command, shell=True, capture_output=True, text= True)
    
    output = result.stdout
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_output = ansi_escape.sub('', output)
        
    pattern = r'i/o =\s*(\d+)/\s*(\d+)\s*lat =\s*(\d+)\s*and =\s*(\d+)\s*lev =\s*(\d+)'
    match = re.search(pattern, clean_output)    
    assert match
    
    pis = int(match.group(1))
    pos = int(match.group(2))
    gates = int(match.group(4))
    lev   = int(match.group(5))

    return pis, pos, gates, lev

def run(root_dir, res_csv, tool_abc):
    abs_aigs_folder = os.path.abspath(root_dir)
    abspath_aigs = glob.glob(os.path.join(abs_aigs_folder, '*.aig'))
    # profile of the aig
    filenames = []
    pis_cnt = []
    pos_cnt = []
    gates_cnt = []
    levs_cnt = []
        
    for aig in abspath_aigs: 
        basename = os.path.basename(aig)
        filename = os.path.splitext(basename)[0]
        pis, pos, gates, lev = apply_abc_profile(aig, tool_abc)
        filenames.append(filename) 
        pis_cnt.append(pis)
        pos_cnt.append(pos)
        gates_cnt.append(gates)
        levs_cnt.append(lev)
     
    # store the profile into csv file          
    data_profile = {
        "filenames":filenames,
        "pis":pis_cnt,
        "pos":pos_cnt,
        "gates":gates_cnt,
        "levels":levs_cnt
    }
    df = pd.DataFrame(data_profile)
    abs_csv_prof = os.path.abspath(res_csv)
    df.to_csv(abs_csv_prof)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Pool AIG data", description="generate the graphml format files")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the aig data directory')
    parser.add_argument('--res_csv', type=str, required=True, help='res csv of profile')
    parser.add_argument('--tool_abc', type=str, required=True, help='abc')
    args = parser.parse_args()
    
    run(args.root_dir, args.res_csv, args.tool_abc)