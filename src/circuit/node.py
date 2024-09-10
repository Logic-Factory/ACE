import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from src.circuit.tag import Tag
from enum import Enum, auto

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
    
    def set_type(self, type_node:str):
        self._type = type_node

    def get_type(self):
        return self._type
    
    def add_fanin(self, fanin:int):
        self._fanins.append(fanin)
    
    def set_fanins(self, fanins:list):
        self._fanins = fanins
    
    def get_fanins(self):
        return self._fanins

    def set_truthtable(self, truthtable:str):
        self._truthtable = truthtable
        
    def get_truthtable(self):
        return self._truthtable

class NodeTypeEnum(Enum):
    """_summary_

    Args:
        enumerate (_type_): _description_
    """
    CONST0 = auto()
    CONST1 = auto()
    PI     = auto()
    PO     = auto()
    # LOGIC GATES
    INVERTER = auto()
    BUFFER   = auto()
    AND2     = auto()
    NAND2    = auto()
    OR2      = auto()
    NOR2     = auto()
    XOR2     = auto()
    XNOR2    = auto()
    MAJ3     = auto()
    XOR3     = auto()
    NAND3    = auto()
    NOR3     = auto()
    MUX21    = auto()
    NMUX21   = auto()
    AOI21    = auto()
    OAI21    = auto()
    AXI21    = auto()
    XAI21    = auto()
    OXI21    = auto()
    XOI21    = auto()
    # TECH GATES
    CELL     = auto()
    
    @staticmethod
    def node_domains():
        domains = {Tag.str_node_const0():NodeTypeEnum.CONST0.value,
                   Tag.str_node_pi():NodeTypeEnum.PI.value,
                   Tag.str_node_po():NodeTypeEnum.PO.value,
                   Tag.str_node_inv():NodeTypeEnum.INVERTER.value,
                   Tag.str_node_and2():NodeTypeEnum.AND2.value,
                   Tag.str_node_nand2():NodeTypeEnum.NAND2.value,
                   Tag.str_node_or2():NodeTypeEnum.OR2.value,
                   Tag.str_node_nor2():NodeTypeEnum.NOR2.value,
                   Tag.str_node_xor2():NodeTypeEnum.XOR2.value,
                   Tag.str_node_xnor2():NodeTypeEnum.XNOR2.value,
                   Tag.str_node_maj3():NodeTypeEnum.MAJ3.value,
                   Tag.str_node_xor3():NodeTypeEnum.XOR3.value,
                   Tag.str_node_nand3():NodeTypeEnum.NAND3.value,
                   Tag.str_node_nor3():NodeTypeEnum.NOR3.value,
                   Tag.str_node_mux21():NodeTypeEnum.MUX21.value,
                   Tag.str_node_nmux21():NodeTypeEnum.NMUX21.value,
                   Tag.str_node_aoi21():NodeTypeEnum.AOI21.value,
                   Tag.str_node_oai21():NodeTypeEnum.OAI21.value,
                   Tag.str_node_axi21():NodeTypeEnum.AXI21.value,
                   Tag.str_node_xai21():NodeTypeEnum.XAI21.value,
                   Tag.str_node_oxi21():NodeTypeEnum.OXI21.value,
                   Tag.str_node_xoi21():NodeTypeEnum.XOI21.value,                   
                   Tag.str_node_cell():NodeTypeEnum.CELL.value
                   }
        return domains
    