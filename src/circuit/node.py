from tag import Tag

class Physics(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Node(object):
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
    CELL = 12
    PO = 13

    def __init__(self, type_node:str, name:str, idx:int, fanins=[], physics:Physics=None):
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
        self._physics = physics
    
    @staticmethod
    def node_domains():
        domains = {Tag.str_node_const0():Node.CONST0,
                   Tag.str_node_pi():Node.PI,
                   Tag.str_node_buffer():Node.BUFFER,
                   Tag.str_node_inverter():Node.INVERTER,
                   Tag.str_node_and2():Node.AND2,
                   Tag.str_node_nand2():Node.NAND2,
                   Tag.str_node_or2():Node.OR2,
                   Tag.str_node_nor2():Node.NOR2,
                   Tag.str_node_xor2():Node.XOR2,
                   Tag.str_node_xnor2():Node.XNOR2,
                   Tag.str_node_maj3():Node.MAJ3,
                   Tag.str_node_xor3():Node.XOR3,
                   Tag.str_node_cell():Node.CELL,
                   Tag.str_node_po():Node.PO}
        return domains

    @staticmethod
    def gate_domains():
        domains = {Tag.str_node_buffer():Node.BUFFER,
                   Tag.str_node_inverter():Node.INVERTER,
                   Tag.str_node_and2():Node.AND2,
                   Tag.str_node_nand2():Node.NAND2,
                   Tag.str_node_or2():Node.OR2,
                   Tag.str_node_nor2():Node.NOR2,
                   Tag.str_node_xor2():Node.XOR2,
                   Tag.str_node_xnor2():Node.XNOR2,
                   Tag.str_node_maj3():Node.MAJ3,
                   Tag.str_node_xor3():Node.XOR3,
                   Tag.str_node_cell():Node.CELL}
        return domains
    
    @staticmethod
    def make_node(type_node, name, idx, fanins):
        if type_node not in Tag.tags_node():
            raise ValueError("Invalid circuit node type")
        return Node(type_node, name, idx, fanins)

    @staticmethod
    def make_gate(type_node, name, idx, fanins):
        if type_node not in Tag.tags_gate():
            raise ValueError("Invalid internal gate type")
        return Node(type_node, name, idx, fanins)    
    
    @staticmethod
    def make_const0(name, idx):
        return Node(Tag.str_node_const0(), name, idx, [])
    
    @staticmethod
    def make_pi(name, idx):
        return Node(Tag.str_node_pi(), name, idx, [])

    @staticmethod
    def make_po(name, idx, fanin0):
        return Node(Tag.str_node_po(), name, idx, [fanin0])    

    @staticmethod
    def make_buffer(name, idx, fanin0):
        return Node(Tag.str_node_buffer(), name, idx,  [fanin0])
    
    @staticmethod
    def make_inverter(name, idx, fanin0):
        return Node(Tag.str_node_inverter(), name, idx,  [fanin0])
    
    @staticmethod
    def make_and2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_and2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_nand2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_nand2(), name, idx,  [fanin0, fanin1])

    @staticmethod
    def make_or2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_or2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_nor2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_nor2(), name, idx,  [fanin0, fanin1])
    
    @staticmethod
    def make_xor2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_xor2(), name, idx,  [fanin0, fanin1])

    @staticmethod
    def make_xnor2(name, idx, fanin0, fanin1):
        return Node(Tag.str_node_xnor2(), name, idx, [fanin0, fanin1])

    @staticmethod
    def make_maj3(name, idx, fanin1, fanin2, fanin3):
        return Node(Tag.str_node_maj3(), name, idx,  [fanin1, fanin2, fanin3] )

    @staticmethod
    def make_xor3(name, idx, fanin1, fanin2, fanin3):
        return Node(Tag.str_node_xor3(), name, idx,  [fanin1, fanin2, fanin3] )

    @staticmethod
    def make_cell(name, idx, fanins:list, physics:Physics=None):
        return Node(Tag.str_node_cell(), name, idx,  fanins, physics )

    def set_idx(self, idx:int):
        self._idx = idx

    def get_idx(self):
        return self._idx
    
    def get_type(self):
        return self._type

    def get_fanins(self):
        return self._fanins
    
    def get_physics(self):
        return self._physics

    def is_const0(self):
        return self._type == Node.CONST0

    def is_pi(self):
        return self._type == Node.PI

    def is_po(self):
        return self._type == Node.PO

    def is_latch(self):
        return self._type == Node.LATCH

    def is_buffer(self):
        return self._type == Node.BUFFER

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

    def is_cell(self):
        return self._type == Node.CELL