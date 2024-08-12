class Tag(object):
    """ Tag the Node and Circuits
    """
    ####################################################
    #   tag of the ciruits
    ####################################################
    @staticmethod
    def tags_circuit():
        tags = ["NONE",
                "GTECH",
                "AIG",
                "XAG",
                "MIG",
                "XMG",
                "CELL" ]
        return tags
    
    @staticmethod
    def str_none():
        return "NONE"

    @staticmethod
    def str_ckt_gtech():
        return "GTECH"
    
    @staticmethod
    def str_ckt_aig():
        return "AIG"
    
    @staticmethod
    def str_ckt_xag():
        return "XAG"
    
    @staticmethod
    def str_ckt_mig():
        return "MIG"
    
    @staticmethod
    def str_ckt_xmg():
        return "XMG"
    
    @staticmethod
    def str_ckt_cell():
        return "CELL"

    ####################################################
    #   tag of the nodes
    ####################################################
    @staticmethod
    def tags_gate():
        tags = ["BUFFER", 
                "INVERTER", 
                "AND2", 
                "NAND2", 
                "OR2", 
                "NOR2", 
                "XOR2", 
                "XNOR2", 
                "MAJ3", 
                "XOR3"]
        return tags

    @staticmethod
    def tags_node():
        tags = ["CONST0", 
                "PI", 
                "PO", 
                "BUFFER", 
                "INVERTER", 
                "AND2", 
                "NAND2", 
                "OR2", 
                "NOR2", 
                "XOR2", 
                "XNOR2", 
                "MAJ3", 
                "XOR3"]
        return tags
    
    @staticmethod
    def str_node_const0():
        return "CONST0"
    
    @staticmethod
    def str_node_pi():
        return "PI"
    
    @staticmethod
    def str_node_po():
        return "PO"
    
    @staticmethod
    def str_node_buffer():
        return "BUFFER"
    
    @staticmethod
    def str_node_inverter():
        return "INVERTER"
    
    @staticmethod
    def str_node_and2():
        return "AND2"
    
    @staticmethod
    def str_node_or2():
        return "OR2"
    
    @staticmethod
    def str_node_xor2():
        return "XOR2"
    
    @staticmethod
    def str_node_nand2():
        return "NAND2"

    @staticmethod
    def str_node_nor2():
        return "NOR2"

    @staticmethod
    def str_node_xnor2():
        return "XNOR2"
    
    @staticmethod
    def str_node_maj3():
        return "MAJ3"
    
    @staticmethod
    def str_node_xor3():
        return "XOR3"
    
    