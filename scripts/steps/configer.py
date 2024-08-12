import os
import argparse
import configparser

class Configer:
    def __init__(self, file):
      self.file = os.path.abspath(file)
      # read the configuraitons
      self.conf = configparser.ConfigParser()
      self.conf.read(self.file, encoding="utf-8")
      
    def get_rtl_tool(self):
        return self.conf['RTL']['synthesis']

    def get_logic_tool(self):
        return self.conf['Logic']['synthesis']

    def get_netlist_tool(self):
        return self.conf['Netlist']['synthesis']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="configer", description="Open Logic Synthesis Dataset")
    parser.add_argument('--file', type=str, required=True, help='path of the confier file')
    args = parser.parse_args()
    
    config = Configer(args.file)