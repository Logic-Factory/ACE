import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from src.circuit.tag import Tag
from enum import Enum

class Node(object):
    """ Logic Node in the Logic circuit graph
    """

    def __init__(self, type_node, idx, fanins, truthtable ):
        """
        :param _type: Type of the node
        :param _idx: Index of the node
        :param _fanins: List of fanins (also in the index format)
        """
        self._type       : str = type_node
        self._idx        : int = idx
        self._fanins     : list[int] = fanins
        self._truthtable : str = truthtable

    # create functions
    @staticmethod
    def make_node(type_node, idx, fanins:list, truthtable:str):
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid circuit node type")
        return Node(type_node, idx, fanins, truthtable)
    
    @staticmethod
    def make_const0(idx, truthtable:str):
        return Node(Tag.str_node_const0(), idx, [], truthtable)

    @staticmethod
    def make_const1(idx, truthtable:str):
        return Node(Tag.str_node_const1(), idx, [], truthtable)
    
    @staticmethod
    def make_pi(idx, truthtable:str):
        return Node(Tag.str_node_pi(), idx, [], truthtable)

    @staticmethod
    def make_po(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_po(), idx, fanins, truthtable)
    
    @staticmethod
    def make_inverter(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_inv(), idx, fanins , truthtable )
    
    @staticmethod
    def make_and2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_and2(), idx, fanins , truthtable )
    
    @staticmethod
    def make_nand2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nand2(), idx, fanins , truthtable )

    @staticmethod
    def make_or2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_or2(), idx, fanins , truthtable )
    
    @staticmethod
    def make_nor2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nor2(), idx, fanins , truthtable )
    @staticmethod
    def make_xor2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xor2(), idx, fanins , truthtable )

    @staticmethod
    def make_xnor2(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xnor2(), idx, fanins , truthtable )

    @staticmethod
    def make_maj3(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_maj3(), idx, fanins , truthtable )

    @staticmethod
    def make_xor3(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xor3(), idx, fanins , truthtable )

    @staticmethod
    def make_nand3(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nand3(), idx, fanins , truthtable )
    
    @staticmethod
    def make_nor3(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nor3(), idx, fanins , truthtable )
    
    @staticmethod
    def make_mux21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_mux21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_nmux21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_nmux21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_aoi21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_aoi21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_oai21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_oai21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_axi21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_axi21(), idx, fanins , truthtable )

    @staticmethod
    def make_xai21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xai21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_oxi21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_oxi21(), idx, fanins , truthtable )
    
    @staticmethod
    def make_xoi21(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_xoi21(), idx, fanins , truthtable )

    @staticmethod
    def make_cell(idx, fanins:list, truthtable:str):
        return Node(Tag.str_node_cell(), idx,  fanins, truthtable )

    # checking functions
    def is_const0(self):
        return self._type == Tag.str_node_const0()
    
    def is_const1(self):
        return self._type == Tag.str_node_const1()

    def is_pi(self):
        return self._type == Tag.str_node_pi()

    def is_po(self):
        return self._type == Tag.str_node_po()

    def is_inverter(self):
        return self._type == Tag.str_node_inv()

    def is_buffer(self):
        return self._type == Tag.str_node_buf()

    def is_and2(self):
        return self._type == Tag.str_node_and2()

    def is_nand2(self):
        return self._type == Tag.str_node_nand2()

    def is_or2(self):
        return self._type == Tag.str_node_or2()

    def is_nor2(self):
        return self._type == Tag.str_node_nor2()

    def is_xor2(self):
        return self._type == Tag.str_node_xor2()

    def is_xnor2(self):
        return self._type == Tag.str_node_xnor2()

    def is_maj3(self):
        return self._type == Tag.str_node_maj3()

    def is_xor3(self):
        return self._type == Tag.str_node_xor3()
    
    def is_nand3(self):
        return self._type == Tag.str_node_nand3()

    def is_nor3(self):
        return self._type == Tag.str_node_nor3()
    
    def is_mux21(self):
        return self._type == Tag.str_node_mux21()

    def is_nmux21(self):
        return self._type == Tag.str_node_nmux21()

    def is_aoi21(self):
        return self._type == Tag.str_node_aoi21()

    def is_oai21(self):
        return self._type == Tag.str_node_oai21()
    
    def is_axi21(self):
        return self._type == Tag.str_node_axi21()

    def is_xai21(self):
        return self._type == Tag.str_node_xai21()
    
    def is_oxi21(self):
        return self._type == Tag.str_node_oxi21()

    def is_xoi21(self):
        return self._type == Tag.str_node_xoi21()

    def is_cell(self):
        return self._type == Tag.str_node_cell()
    
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

class NodeTypeEnum(Enum):
    """_summary_

    Args:
        enumerate (_type_): _description_
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
    
    @staticmethod
    def node_domains():
        domains = {Tag.str_node_const0():NodeTypeEnum.CONST0,
                   Tag.str_node_pi():NodeTypeEnum.PI,
                   Tag.str_node_po():NodeTypeEnum.PO,
                   Tag.str_node_inv():NodeTypeEnum.INVERTER,
                   Tag.str_node_and2():NodeTypeEnum.AND2,
                   Tag.str_node_nand2():NodeTypeEnum.NAND2,
                   Tag.str_node_or2():NodeTypeEnum.OR2,
                   Tag.str_node_nor2():NodeTypeEnum.NOR2,
                   Tag.str_node_xor2():NodeTypeEnum.XOR2,
                   Tag.str_node_xnor2():NodeTypeEnum.XNOR2,
                   Tag.str_node_maj3():NodeTypeEnum.MAJ3,
                   Tag.str_node_xor3():NodeTypeEnum.XOR3,
                   Tag.str_node_nand3():NodeTypeEnum.NAND3,
                   Tag.str_node_nor3():NodeTypeEnum.NOR3,
                   Tag.str_node_mux21():NodeTypeEnum.MUX21,
                   Tag.str_node_nmux21():NodeTypeEnum.NMUX21,
                   Tag.str_node_aoi21():NodeTypeEnum.AOI21,
                   Tag.str_node_oai21():NodeTypeEnum.OAI21,
                   Tag.str_node_axi21():NodeTypeEnum.AXI21,
                   Tag.str_node_xai21():NodeTypeEnum.XAI21,
                   Tag.str_node_oxi21():NodeTypeEnum.OXI21,
                   Tag.str_node_xoi21():NodeTypeEnum.XOI21,                   
                   Tag.str_node_cell():NodeTypeEnum.CELL
                   }
        return domains
    