import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

###########################################################
#                   gate simulation
# @note:
#   1. all the gates are based on the boolean logic
#   2. the input and output are all boolean
#   3. the inputs are followed the order by high to low
###########################################################
def sim_buffer(sig1:bool):
    return sig1

def sim_inverter(sig1:bool):
    return not sig1
    
def sim_and2(sig1:bool, sig2:bool):
    return sig1 and sig2

def sim_nand2(sig1:bool, sig2:bool):
    return not (sig1 and sig2)

def sim_or2(sig1:bool, sig2:bool):
    return sig1 or sig2

def sim_nor2(sig1:bool, sig2:bool):
    return not (sig1 or sig2)

def sim_xor2(sig1:bool, sig2:bool):
    return sig1 != sig2

def sim_xnor2(sig1:bool, sig2:bool):
    return sig1 == sig2

def sim_maj3(sig1:bool, sig2:bool, sig3:bool):
    return (sig1 and sig2) or (sig1 and sig3) or (sig2 and sig3)

def sim_xor3(sig1:bool, sig2:bool, sig3:bool):
    return sig1 != sig2 != sig3

def sim_nand3(sig1:bool, sig2:bool, sig3:bool):
    return not (sig1 and sig2 and sig3)

def sim_nor3(sig1:bool, sig2:bool, sig3:bool):
    return not (sig1 or sig2 or sig3)

def sim_mux21(sel:bool, sig1:bool, sig2:bool):
    return sig1 if sel else sig2

def sim_nmux21(sel:bool, sig1:bool, sig2:bool):
    return sig2 if sel else sig1

def sim_aoi21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel and sig1) or sig2 )

def sim_oai21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel or sig1) and sig2 )

def sim_axi21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel != sig1) or sig2 )

def sim_xai21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel != sig1) and sig2 )

def sim_oxi21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel and sig1) or sig2 )

def sim_xoi21(sel:bool, sig1:bool, sig2:bool):
    return not ( (sel != sig1) and sig2 )

def sim_cell(tt:str, *args:list[bool]):
    int_val1 = int(tt[-16:], 16)
    index = 0
    for i, b in enumerate( reversed(args) ):
        if b:
            index += 2**i
    return (int_val1 >> index) & 1

###########################################################
#             truthtable operation
# @note:
#   1. 
###########################################################
def compute_tt_hex_and2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = int_val1 & int_val2
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_nand2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = ~(int_val1 & int_val2)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_or2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = int_val1 | int_val2
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_nor2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = ~(int_val1 | int_val2)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_xor2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = int_val1 ^ int_val2
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_xnor2(tt1:str, tt2:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = ~(int_val1 ^ int_val2)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_maj3(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = (int_val1 & int_val2) | (int_val1 & int_val3) | (int_val2 & int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_xor3(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = int_val1 ^ int_val2 ^ int_val3
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_nand3(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~(int_val1 & int_val2 & int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_nor3(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~(int_val1 | int_val2 | int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_mux21(sel:str, tt1:str, tt2:str):
    int_vals = int(sel[-16:], 16)
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = 0
    for i in range(64):
        if int_vals & (1 << i):
            int_res |= int_val1 & (1 << i)
        else:
            int_res |= int_val2 & (1 << i)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_nmux(sel:str, tt1:str, tt2:str):
    int_vals = int(sel[-16:], 16)
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_res = 0
    for i in range(64):
        if int_vals & (1 << i):
            int_res |= int_val2 & (1 << i)
        else:
            int_res |= int_val1 & (1 << i)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_aoi21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 & int_val2) | int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_oai21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 | int_val2) & int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_axi21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 & int_val2) ^ int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_xai21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 ^ int_val2) & int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_oxi21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 | int_val2) ^ int_val3)
    hex_res = hex(int_res)
    return hex_res

def compute_tt_hex_xoi21(tt1:str, tt2:str, tt3:str):
    int_val1 = int(tt1[-16:], 16)
    int_val2 = int(tt2[-16:], 16)
    int_val3 = int(tt3[-16:], 16)
    int_res = ~( (int_val1 ^ int_val2) | int_val3)
    hex_res = hex(int_res)
    return hex_res

if __name__ == "__main__":
    assert sim_buffer(True) == True
    assert sim_buffer(False) == False
    assert sim_inverter(True) == False
    assert sim_inverter(False) == True
    assert sim_and2(True, True) == True
    assert sim_and2(True, False) == False
    assert sim_and2(False, True) == False
    assert sim_and2(False, False) == False
    
    pass

