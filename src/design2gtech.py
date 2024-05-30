import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class DesignToGtech:
    def __init__(self, root_dir, data_type, yosys_tool_path, gtechlib_path, res_dir):
        self.root_dir = root_dir
        self.data_type = data_type
        self.yosys_tool_path = yosys_tool_path
        self.gtechlib_path = gtechlib_path
        self.res_dir = res_dir


    def run(self):
        # Select the appropriate file type conversion based on the data type
        if self.data_type == 'aig':
            aig_files = [f for f in os.listdir(self.root_dir) if f.endswith('.aig')]
            with ThreadPoolExecutor(max_workers = 4) as executor:
                futures = []
                total_tasks = len(aig_files)
                for aig_file in aig_files:
                    aig_file_path = os.path.join(self.root_dir, aig_file)
                    gtech_file_path = os.path.join(self.res_dir ,aig_file + '_gtech.v')
                    future = executor.submit(self.aig_to_gtech, aig_file_path, gtech_file_path)
                    futures.append(future)
                
                for future in tqdm(futures, total=total_tasks):
                    future.result()
            print('AIG to Gtech conversion completed.')
        
        elif self.data_type == 'verilog':
            verilog_files = [f for f in os.listdir(self.root_dir) if f.endswith('.v')]
            with ThreadPoolExecutor(max_workers = 4) as executor:
                futures = []
                total_tasks = len(verilog_files)
                for verilog_file in verilog_files:
                    verilog_file_path = os.path.join(self.root_dir, verilog_file)
                    gtech_file_path = os.path.join(self.res_dir ,verilog_file + '_gtech.v')
                    future = executor.submit(self.verilog_to_gtech, verilog_file_path,gtech_file_path)
                    futures.append(future)

                for future in tqdm(futures, total=total_tasks):
                    future.result()
            print('Verilog to Gtech conversion completed.')

        else:
            print('Unsupported data type.')


    def aig_to_gtech(self, aig_in, gtech_out):
        """Invoke the yosys tool to convert AIG to Gtech format

        Args:
            aig_in (str): Path to the AIG file
            gtech_out (str): Path to the output Gtech file
        """
        # read_aiger ../benchmark/kdd24/adder.aig
        # aigmap
        # techmap
        # abc -genlib ../techlib/gtech.genlib
        # write_verilog adder_gtech.v

        script = "read_aiger {0}; aigmap; techmap; abc -genlib {1}; write_verilog {2}"\
            .format(aig_in, self.gtechlib_path, gtech_out)
        command = "{0} -p {1}".format(self.yosys_tool_path, script)
        subprocess.run(command, shell=True, capture_output=True, text= True)


    def verilog_to_gtech(self, verilog_in, gtech_out):
        """Invoke the yosys tool to convert Verilog to Gtech format

        Args:
            verilog_in (str): Path to the Verilog file
            gtech_out (str): Path to the output Gtech file
        """
        # read_verilog ../benchmark/kdd24/adder.v; 
        # hierarchy -check; 
        # proc; fsm; memory; 
        # techmap; 
        # abc -genlib ../techlib/gtech.genlib; 
        # write_verilog adder_gtech.v

        script = "read_verilog {0}; hierarchy -check; proc; fsm; memory; techmap; abc -genlib {1}; \
            write_verilog {2}".format(verilog_in, self.gtechlib_path, gtech_out)
        command = "{0} -p {1}".format(self.yosys_tool_path, script)
        subprocess.run(command, shell=True, capture_output=True, text= True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert design files to Gtech format.')
    parser.add_argument('--root_dir', type=str,required=True, help='Root directory of design files.')
    parser.add_argument('--data_type', type=str,default='verilog', required=True, help='Data type of design files. Choose from "verilog" or "aig".')
    parser.add_argument('--yosys_tool_path', type=str, required=True, help='Path to yosys tool.')
    parser.add_argument('--gtechlib_path', type=str, default='../techlib/gtech.genlib',required=True, help='Path to Gtech library.')
    parser.add_argument('--res_dir', type=str,default='../results', required=True, help='Path to save the converted files.')
    args = parser.parse_args()

    design_to_gtech = DesignToGtech(args.root_dir, args.data_type, args.yosys_tool_path, args.gtechlib_path, args.res_dir)
    design_to_gtech.run()