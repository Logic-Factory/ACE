import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from src.circuit.tag import Tag

class Node(object):
    """ Logic Node in the Logic circuit graph
    """
    CONST0 = 0
    CONST1 = 1
    PI = 2
    PO = 3
    # LOGIC GATES
    INVERTER = 4
    AND2 = 5
    NAND2 = 6
    OR2 = 7
    NOR2 = 8
    XOR2 = 9
    XNOR2 = 10
    MAJ3 = 11
    XOR3 = 12
    NAND3 = 13
    NOR3 = 14
    MUX21 = 15
    NMUX21 = 16
    AOI21 = 17
    OAI21 = 18
    AXI21 = 19
    XAI21 = 20
    OXI21 = 21
    XOI21 = 22
    # TECH GATES
    CELL = 23

    def __init__(self, type_node:str, name:str, idx:int, fanins:list, truthtable:str ):
        """
        :param _type: Type of the node
        :param _name: Name of the node, this is corresponding to the original node
        :param _idx: Index of the node
        :param _fanins: List of fanins (also in the index format)
        """
        self._type = type_node
        self._name = name
        self._idx = idx
        self._fanins = fanins
        self._truthtable = truthtable
    
    @staticmethod
    def node_domains():
        domains = {Tag.str_node_const0():Node.CONST0,
                   Tag.str_node_pi():Node.PI,
                   Tag.str_node_po():Node.PO,
                   Tag.str_node_inv():Node.INVERTER,
                   Tag.str_node_and2():Node.AND2,
                   Tag.str_node_nand2():Node.NAND2,
                   Tag.str_node_or2():Node.OR2,
                   Tag.str_node_nor2():Node.NOR2,
                   Tag.str_node_xor2():Node.XOR2,
                   Tag.str_node_xnor2():Node.XNOR2,
                   Tag.str_node_maj3():Node.MAJ3,
                   Tag.str_node_xor3():Node.XOR3,
                   Tag.str_node_nand3():Node.NAND3,
                   Tag.str_node_nor3():Node.NOR3,
                   Tag.str_node_mux21():Node.MUX21,
                   Tag.str_node_nmux21():Node.NMUX21,
                   Tag.str_node_aoi21():Node.AOI21,
                   Tag.str_node_oai21():Node.OAI21,
                   Tag.str_node_axi21():Node.AXI21,
                   Tag.str_node_xai21():Node.XAI21,
                   Tag.str_node_oxi21():Node.OXI21,
                   Tag.str_node_xoi21():Node.XOI21,                   
                   Tag.str_node_cell():Node.CELL
                   }
        return domains

    @staticmethod
    def gate_domains():
        domains = {Tag.str_node_inv():Node.INVERTER,
                   Tag.str_node_and2():Node.AND2,
                   Tag.str_node_nand2():Node.NAND2,
                   Tag.str_node_or2():Node.OR2,
                   Tag.str_node_nor2():Node.NOR2,
                   Tag.str_node_xor2():Node.XOR2,
                   Tag.str_node_xnor2():Node.XNOR2,
                   Tag.str_node_maj3():Node.MAJ3,
                   Tag.str_node_xor3():Node.XOR3,
                   Tag.str_node_nand3():Node.NAND3,
                   Tag.str_node_nor3():Node.NOR3,
                   Tag.str_node_mux21():Node.MUX21,
                   Tag.str_node_nmux21():Node.NMUX21,
                   Tag.str_node_aoi21():Node.AOI21,
                   Tag.str_node_oai21():Node.OAI21,
                   Tag.str_node_axi21():Node.AXI21,
                   Tag.str_node_xai21():Node.XAI21,
                   Tag.str_node_oxi21():Node.OXI21,
                   Tag.str_node_xoi21():Node.XOI21,                   
                   Tag.str_node_cell():Node.CELL
                   }
        return domains
    
    # create functions
    @staticmethod
    def make_node(type_node, name, idx, fanins:list, truthtable:str):
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid circuit node type")
        return Node(type_node, name, idx, fanins)

    @staticmethod
    def make_gate(type_node, name, idx, fanins:list, truthtable:str):
        if type_node not in Tag.tags_gate():
            raise ValueError("Invalid internal gate type")
        return Node(type_node, name, idx, fanins)    
    
    @staticmethod
    def make_const0(name, idx, truthtable:str):
        return Node(Tag.str_node_const0(), name, idx, [], truthtable)

    @staticmethod
    def make_const1(name, idx, truthtable:str):
        return Node(Tag.str_node_const1(), name, idx, [], truthtable)
    
    @staticmethod
    def make_pi(name, idx, truthtable:str):
        return Node(Tag.str_node_pi(), name, idx, [], truthtable)

    @staticmethod
    def make_po(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_po(), name, idx, fanins, truthtable)
    
    @staticmethod
    def make_inverter(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_inv(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_and2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_and2(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_nand2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nand2(), name, idx, fanins , truthtable )

    @staticmethod
    def make_or2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_or2(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_nor2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nor2(), name, idx, fanins , truthtable )
    @staticmethod
    def make_xor2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xor2(), name, idx, fanins , truthtable )

    @staticmethod
    def make_xnor2(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xnor2(), name, idx, fanins , truthtable )

    @staticmethod
    def make_maj3(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_maj3(), name, idx, fanins , truthtable )

    @staticmethod
    def make_xor3(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xor3(), name, idx, fanins , truthtable )

    @staticmethod
    def make_nand3(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nand3(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_nor3(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nor3(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_mux21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_mux21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_nmux21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nmux21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_aoi21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_aoi21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_oai21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_oai21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_axi21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_axi21(), name, idx, fanins , truthtable )

    @staticmethod
    def make_xai21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xai21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_oxi21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_oxi21(), name, idx, fanins , truthtable )
    
    @staticmethod
    def make_xoi21(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xoi21(), name, idx, fanins , truthtable )

    @staticmethod
    def make_cell(name, idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_cell(), name, idx,  fanins, truthtable )

    # checking functions
    def is_const0(self):
        return self._type == Node.CONST0
    
    def is_const1(self):
        return self._type == Node.CONST1

    def is_pi(self):
        return self._type == Node.PI

    def is_po(self):
        return self._type == Node.PO

    def is_inverter(self):
        return self._type == Node.INVERTER

    def is_and2(self):
        return self._type == Node.AND2

    def is_nand2(self):
        return self._type == Node.NAND2

    def is_or2(self):
        return self._type == Node.OR2

    def is_nor2(self):
        return self._type == Node.NOR2

    def is_xor2(self):
        return self._type == Node.XOR2

    def is_xnor2(self):
        return self._type == Node.XNOR2

    def is_maj3(self):
        return self._type == Node.MAJ3

    def is_xor3(self):
        return self._type == Node.XOR3
    
    def is_nand3(self):
        return self._type == Node.NAND3

    def is_nor3(self):
        return self._type == Node.NOR3
    
    def is_mux21(self):
        return self._type == Node.MUX21

    def is_nmux21(self):
        return self._type == Node.NMUX21

    def is_aoi21(self):
        return self._type == Node.AOI21

    def is_oai21(self):
        return self._type == Node.OAI21
    
    def is_axi21(self):
        return self._type == Node.AXI21

    def is_xai21(self):
        return self._type == Node.XAI21
    
    def is_oxi21(self):
        return self._type == Node.OXI21

    def is_xoi21(self):
        return self._type == Node.XOI21

    def is_cell(self):
        return self._type == Node.CELL
    
    # misc function
    def set_idx(self, idx:int):
        self._idx = idx

    def get_idx(self):
        return self._idx
    
    def get_type(self):
        return self._type

    def add_fanin(self, fanin:int):
        self._fanins.append(fanin)

    def get_fanins(self):
        return self._fanins
    
    def get_physics(self):
        return self._physics