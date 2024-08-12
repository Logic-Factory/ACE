from tag import Tag

class LogicNode(object):
    """ Logic Node in the Logic circuit graph
    """
    CONST0 = 0
    PI = 1
    BUFFER = 2
    INVERTER = 3
    AND2 = 4
    NAND2 = 5
    OR2 = 6
    NOR2 = 7
    XOR2 = 8
    XNOR2 = 9
    MAJ3 = 10
    XOR3 = 11
    PO = 12

    def __init__(self, type_node:str, name:str, idx:int, fanins=[]):
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
    
    @staticmethod
    def node_domains():
        domains = {Tag.str_node_const0():LogicNode.CONST0,
                   Tag.str_node_pi():LogicNode.PI,
                   Tag.str_node_buffer():LogicNode.BUFFER,
                   Tag.str_node_inverter():LogicNode.INVERTER,
                   Tag.str_node_and2():LogicNode.AND2,
                   Tag.str_node_nand2():LogicNode.NAND2,
                   Tag.str_node_or2():LogicNode.OR2,
                   Tag.str_node_nor2():LogicNode.NOR2,
                   Tag.str_node_xor2():LogicNode.XOR2,
                   Tag.str_node_xnor2():LogicNode.XNOR2,
                   Tag.str_node_maj3():LogicNode.MAJ3,
                   Tag.str_node_xor3():LogicNode.XOR3,
                   Tag.str_node_po():LogicNode.PO}
        return domains

    @staticmethod
    def gate_domains():
        domains = {Tag.str_node_buffer():LogicNode.BUFFER,
                   Tag.str_node_inverter():LogicNode.INVERTER,
                   Tag.str_node_and2():LogicNode.AND2,
                   Tag.str_node_nand2():LogicNode.NAND2,
                   Tag.str_node_or2():LogicNode.OR2,
                   Tag.str_node_nor2():LogicNode.NOR2,
                   Tag.str_node_xor2():LogicNode.XOR2,
                   Tag.str_node_xnor2():LogicNode.XNOR2,
                   Tag.str_node_maj3():LogicNode.MAJ3,
                   Tag.str_node_xor3():LogicNode.XOR3}
        return domains
    
    @staticmethod
    def make_node(type_node, name, idx, fanins):
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid circuit node type")
        return LogicNode(type_node, name, idx, fanins)

    @staticmethod
    def make_gate(type_node, name, idx, fanins):
        if type_node not in Tag.tags_gate():
            raise ValueError("Invalid internal gate type")
        return LogicNode(type_node, name, idx, fanins)    
    
    @staticmethod
    def make_const0(name, idx):
        return LogicNode(Tag.str_node_const0(), name, idx, [])
    
    @staticmethod
    def make_pi(name, idx):
        return LogicNode(Tag.str_node_pi(), name, idx, [])

    @staticmethod
    def make_po(name, idx, fanin0):
        return LogicNode(Tag.str_node_po(), name, idx, [fanin0])    

    @staticmethod
    def make_buffer(name, idx, fanin0):
        return LogicNode(Tag.str_node_buffer(), name, idx,  [fanin0])
    
    @staticmethod
    def make_inverter(name, idx, fanin0):
        return LogicNode(Tag.str_node_inverter(), name, idx,  [fanin0])
    
    @staticmethod
    def make_and2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_and2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_nand2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_nand2(), name, idx,  [fanin0, fanin1])

    @staticmethod
    def make_or2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_or2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_nor2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_nor2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_xor2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_xor2(), name, idx,  [fanin0, fanin1])

    @staticmethod
    def make_xnor2(name, idx, fanin0, fanin1):
        return LogicNode(Tag.str_node_xnor2(), name, idx, [fanin0, fanin1])

    @staticmethod
    def make_maj3(name, idx, fanin1, fanin2, fanin3):
        return LogicNode(Tag.str_node_maj3(), name, idx,  [fanin1, fanin2, fanin3] )

    @staticmethod
    def make_xor3(name, idx, fanin1, fanin2, fanin3):
        return LogicNode(Tag.str_node_xor3(), name, idx,  [fanin1, fanin2, fanin3] )

    def set_idx(self, idx:int):
        self._idx = idx

    def get_idx(self):
        return self._idx
    
    def get_type(self):
        return self._type

    def get_fanins(self):
        return self._fanins

    def is_const0(self):
        return self._type == LogicNode.CONST0

    def is_pi(self):
        return self._type == LogicNode.PI

    def is_po(self):
        return self._type == LogicNode.PO

    def is_latch(self):
        return self._type == LogicNode.LATCH

    def is_buffer(self):
        return self._type == LogicNode.BUFFER

    def is_inverter(self):
        return self._type == LogicNode.INVERTER

    def is_and2(self):
        return self._type == LogicNode.AND2

    def is_nand2(self):
        return self._type == LogicNode.NAND2

    def is_or2(self):
        return self._type == LogicNode.OR2

    def is_nor2(self):
        return self._type == LogicNode.NOR2

    def is_xor2(self):
        return self._type == LogicNode.XOR2

    def is_xnor2(self):
        return self._type == LogicNode.XNOR2

    def is_maj3(self):
        return self._type == LogicNode.MAJ3

    def is_xor3(self):
        return self._type == LogicNode.XOR3