class Tag(object):
    """ Tag the Node and Circuits
    """
    ####################################################
    #   tag of the ciruits
    ####################################################
    @staticmethod
    def tags_circuit():
        tags = [ Tag.str_ckt_abc(),
                Tag.str_ckt_aig(),
                Tag.str_ckt_aog(),
                Tag.str_ckt_oig(),
                Tag.str_ckt_xag(),
                Tag.str_ckt_xog(),
                Tag.str_ckt_mig(),
                Tag.str_ckt_xmg(),
                Tag.str_ckt_primary(),
                Tag.str_ckt_gtech(),
                Tag.str_ckt_cell() ]
        return tags

    @staticmethod
    def str_ckt_abc():
        return "ABC"

    @staticmethod
    def str_ckt_aig():
        return "AIG"
    
    @staticmethod
    def str_ckt_aog():
        return "AOG"
    
    @staticmethod
    def str_ckt_oig():
        return "OIG"
    
    @staticmethod
    def str_ckt_xag():
        return "XAG"

    @staticmethod
    def str_ckt_xog():
        return "XOG"    
    
    @staticmethod
    def str_ckt_mig():
        return "MIG"
    
    @staticmethod
    def str_ckt_xmg():
        return "XMG"
    
    @staticmethod
    def str_ckt_primary():
        return "PRIMARY"
    
    @staticmethod
    def str_ckt_gtech():
        return "GTG"
    
    @staticmethod
    def str_ckt_cell():
        return "CELL"

    ####################################################
    #   tag of the nodes
    ####################################################
    @staticmethod
    def tags_node():
        tags = [ Tag.str_node_const0(),
                 Tag.str_node_const1(),
                 Tag.str_node_pi(),
                 Tag.str_node_po()]
        tags.extend(Tag.tags_gate())
        return tags
    
    @staticmethod
    def tags_gate():
        tags = [ Tag.str_node_inv(), 
                 Tag.str_node_and2(),
                 Tag.str_node_nand2(),
                 Tag.str_node_or2(),
                 Tag.str_node_nor2(),
                 Tag.str_node_xor2(),
                 Tag.str_node_xnor2(),
                 Tag.str_node_maj3(),
                 Tag.str_node_xor3(),
                 Tag.str_node_nand3(),
                 Tag.str_node_nor3(),
                 Tag.str_node_mux21(),
                 Tag.str_node_nmux21(),
                 Tag.str_node_aoi21(),
                 Tag.str_node_oai21(),
                 Tag.str_node_axi21(),
                 Tag.str_node_xai21(),
                 Tag.str_node_oxi21(),
                 Tag.str_node_xoi21(),
                 Tag.str_node_cell()]
        return tags

    @staticmethod
    def str_node_const0():
        return "GTECH_CONST0"

    @staticmethod
    def str_all_const0():
        return ["GTECH_CONST0", "_const0_"]

    @staticmethod
    def str_node_const1():
        return "GTECH_CONST1"

    @staticmethod
    def str_all_const1():
        return ["GTECH_CONST1", "_const1_"]
    
    @staticmethod
    def str_node_pi():
        return "GTECH_PI"
    
    @staticmethod
    def str_node_po():
        return "GTECH_PO"

    @staticmethod
    def str_node_inv():
        return "GTECH_INV"

    @staticmethod
    def str_node_buf():
        return "GTECH_BUF"
    
    @staticmethod
    def str_node_and2():
        return "GTECH_AND2"
    
    @staticmethod
    def str_node_or2():
        return "GTECH_OR2"
    
    @staticmethod
    def str_node_xor2():
        return "GTECH_XOR2"
    
    @staticmethod
    def str_node_nand2():
        return "GTECH_NAND2"

    @staticmethod
    def str_node_nor2():
        return "GTECH_NOR2"

    @staticmethod
    def str_node_xnor2():
        return "GTECH_XNOR2"
    
    @staticmethod
    def str_node_maj3():
        return "GTECH_MAJ3"
    
    @staticmethod
    def str_node_xor3():
        return "GTECH_XOR3"

    @staticmethod
    def str_node_nand3():
        return "GTECH_NAND3"

    @staticmethod
    def str_node_nor3():
        return "GTECH_NOR3"

    @staticmethod
    def str_node_mux21():
        return "GTECH_MUX21"

    @staticmethod
    def str_node_nmux21():
        return "GTECH_NMUX21"

    @staticmethod
    def str_node_aoi21():
        return "GTECH_AOI21"

    @staticmethod
    def str_node_oai21():
        return "GTECH_OAI21"

    @staticmethod
    def str_node_axi21():
        return "GTECH_AXI21"

    @staticmethod
    def str_node_xai21():
        return "GTECH_XAI21"

    @staticmethod
    def str_node_oxi21():
        return "GTECH_OXI21"

    @staticmethod
    def str_node_xoi21():
        return "GTECH_XOI21"

    @staticmethod
    def str_node_cell():
        return "GTECH_CELL"