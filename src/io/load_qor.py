import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import json
import gzip

class QoR(object):
    def __init__(self):
        self.size = 0
        self.depth = 0
        self.area = 0
        self.delay = 0
        self.timing = 0
        self.power_total = 0
        self.power_internal = 0
        self.power_leakage = 0
        self.power_dynamic = 0
    
    def set_size(self, size):
        self.size = size

    def set_depth(self, depth):
        self.depth = depth
        
    def set_area(self, area):
        self.area = area
    
    def set_delay(self, delay):
        self.delay = delay
    
    def set_timing(self, timing):
        self.timing = timing
    
    def set_power_total(self, power_total):
        self.power_total = power_total
    
    def set_power_internal(self, power_internal):
        self.power_internal = power_internal
        
    def set_power_leakage(self, power_leakage):
        self.power_leakage = power_leakage
    
    def set_power_dynamic(self, power_dynamic):
        self.power_dynamic = power_dynamic
    
    def get_size(self):
        return self.size

    def get_depth(self):
        return self.depth

    def get_area(self):
        return self.area

    def get_delay(self):
        return self.delay

    def get_timing(self):
        return self.timing

    def get_power_total(self):
        return self.power_total

    def get_power_internal(self):
        return self.power_internal

    def get_power_leakage(self):
        return self.power_leakage

    def get_power_dynamic(self):
        return self.power_dynamic

def load_qor(file):
    count = 0
    if file.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(file, 'rt') as f:
        data = json.load(f)
        qor = QoR()
        if 'gates' in data:
            qor.set_size(data['gates'])
            count += 1
        if 'depth' in data:
            qor.set_depth(data['depth'])
            count += 1
        if 'area' in data:
            qor.set_area(data['area'])
            count += 1
        if 'delay' in data:
            qor.set_delay(data['delay'])
            count += 1
        if 'arrive_time' in data:
            qor.set_timing(data['arrive_time'])
            count += 1
        if 'total_power' in data:
            qor.set_power_total(data['total_power'])
            count += 1
        if 'internal_power' in data:
            qor.set_power_internal(data['internal_power'])
            count += 1
        if 'leakage_power' in data:
            qor.set_power_leakage(data['leakage_power'])
            count += 1
        if 'dynamic_power' in data:
            qor.set_power_dynamic(data['dynamic_power'])
            count += 1
        if count == 0:
            raise Exception('No QoR data in file')
        return qor
    
if __name__ == '__main__':

    file = sys.argv[1]
    qor = load_qor(file)
    
    print('Size: {}'.format(qor.get_size()))
    print('Depth: {}'.format(qor.get_depth()))
    print('Area: {}'.format(qor.get_area()))
    print('Delay: {}'.format(qor.get_delay()))
    print('Timing: {}'.format(qor.get_timing()))
    print('Power Total: {}'.format(qor.get_power_total()))
    print('Power Internal: {}'.format(qor.get_power_internal()))
    print('Power Leakage: {}'.format(qor.get_power_leakage()))
    print('Power Dynamic: {}'.format(qor.get_power_dynamic()))
    