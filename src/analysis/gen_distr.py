import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import glob
import re
import argparse
import matplotlib.pyplot as plt
from src.io.load_qor import load_qor, QoR


def sort_by_recipe_number(file_path):
    match = re.search(r'recipe_(\d+)', os.path.basename(file_path))
    if match:
        return int(match.group(1))
    else:
        return float('inf')

def gen_one_repre_distr(folder):
    qor_files_logic = glob.glob(os.path.join(folder, '*.logic.qor.json'))
    qor_files_asic = glob.glob(os.path.join(folder, '*.asic.qor.json'))
    qor_files_physics = glob.glob(os.path.join(folder, '*.physics.qor.json'))
    qor_files_fpga = glob.glob(os.path.join(folder, '*.fpga.qor.json'))
    
    # keep the same order
    qor_files_logic.sort(key=sort_by_recipe_number)
    qor_files_asic.sort(key=sort_by_recipe_number)
    qor_files_physics.sort(key=sort_by_recipe_number)
    qor_files_fpga.sort(key=sort_by_recipe_number)
    
    # ensure all the qor generated
    if not (len(qor_files_logic) == len(qor_files_asic) == len(qor_files_physics) == len(qor_files_fpga)):
        print('Not all the qor generated')
        print('logic:', len(qor_files_logic))
        print(qor_files_logic)
        print('asic:', len(qor_files_asic))
        print(qor_files_asic)
        print('physics:', len(qor_files_physics))
        print(qor_files_physics)
        print('fpga:', len(qor_files_fpga))
        print(qor_files_fpga)
    assert len(qor_files_logic) == len(qor_files_asic) == len(qor_files_physics) == len(qor_files_fpga)
    
    area_logic  = []
    delay_logic = []
    area_asic  = []
    delay_asic = []
    area_fpga  = []
    delay_fpga = []
    
    for file in qor_files_logic:
        qor = load_qor(file)
        area_logic.append(qor.get_size())
        delay_logic.append(qor.get_delay())
    
    for file in qor_files_asic:
        qor = load_qor(file)
        area_asic.append(qor.get_area())

    for file in qor_files_physics:
        qor = load_qor(file)
        delay_asic.append(qor.get_timing())
        
    for file in qor_files_fpga:
        qor = load_qor(file)
        area_fpga.append(qor.get_area())
        delay_fpga.append(qor.get_delay())

    # plot the area and delay distribution for logic, asic, and fpga by points
    plt.scatter(area_logic, delay_logic, alpha=0.5, label='area vs. delay')
    plt.xlabel('Area')
    plt.ylabel('Delay')
    plt.savefig(os.path.join(folder, 'qor_distr_logic.pdf'))
    plt.clf()
    plt.close()

    plt.scatter(area_fpga, delay_fpga, alpha=0.5, label='area vs. delay')
    plt.xlabel('Area')
    plt.ylabel('Delay')
    plt.savefig(os.path.join(folder, 'qor_distr_fpga.pdf'))
    plt.clf()
    plt.close()

    plt.scatter(area_asic, delay_asic, alpha=0.5, label='area vs. delay')
    plt.xlabel('Area')
    plt.ylabel('Delay')
    plt.savefig(os.path.join(folder, 'qor_distr_asic.pdf'))
    plt.clf()
    plt.close()
    return area_logic, delay_logic, area_asic, delay_asic, area_fpga, delay_fpga

def gen_one_design_distr(folder):
    loigcs = ["abc", "aig", "oig", "aog", "xag", "xog", "primary", "mig", "xmg", "gtg"]

    qor_data = {}
    color_list = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'black', 'gray', 'orange']
    
    for logic in loigcs:
        print(logic)
        area_logic, delay_logic, area_asic, delay_asic, area_fpga, delay_fpga = gen_one_repre_distr(os.path.join(folder, logic))
        qor_data[logic] = {
            "area_logic": area_logic,
            "delay_logic": delay_logic,
            "area_asic": area_asic,
            "delay_asic": delay_asic,
            "area_fpga": area_fpga,
            "delay_fpga": delay_fpga
        }
    # compare the area and delay distribution for logic, asic, and fpga by points with different color
    for i in range( len(qor_data["abc"]) ):
        area_logic = []
        delay_logic = []
        area_asic = []
        delay_asic = []
        area_fpga = []
        delay_fpga = []
        for logic in loigcs:
            area_logic.append(qor_data[logic]["area_logic"][i])
            delay_logic.append(qor_data[logic]["delay_logic"][i])
            area_asic.append(qor_data[logic]["area_asic"][i])
            delay_asic.append(qor_data[logic]["delay_asic"][i])
            area_fpga.append(qor_data[logic]["area_fpga"][i])
            delay_fpga.append(qor_data[logic]["delay_fpga"][i])
            
        # plt the points
        plt.scatter(area_logic, delay_logic, color = color_list, alpha=0.5, label='logic representation')
        plt.xlabel('Area')
        plt.ylabel('Delay')
        plt.legend()
        plt.savefig(os.path.join(folder, 'qor_distr_'+str(i)+'_logic.pdf'))
        plt.clf()
        plt.close()

        # plt the points
        plt.scatter(area_asic, delay_asic, color = color_list, alpha=0.5, label='logic representation')
        plt.xlabel('Area')
        plt.ylabel('Delay')
        plt.savefig(os.path.join(folder, 'qor_distr_'+str(i)+'_asic.pdf'))
        plt.clf()
        plt.close()

        # plt the points
        plt.scatter(area_fpga, delay_fpga, color = color_list, alpha=0.5, label='logic representation')
        plt.xlabel('Area')
        plt.ylabel('Delay')
        plt.savefig(os.path.join(folder, 'qor_distr_'+str(i)+'_fpga.pdf'))
        plt.clf()
        plt.close()
        
def gen_designs(folder):
    # current folder's subfolder
    for subfolder in os.listdir(folder):
        print( "process at :", subfolder)
        gen_one_design_distr(os.path.join(folder, subfolder))
        

if __name__ == '__main__':
    folder = sys.argv[1]
    gen_designs(folder)