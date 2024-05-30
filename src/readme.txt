Step1: Genegate the gtech file by Yosys:
    AIG type:
        read_aiger <aig_file>
        aig_map
        techmap
        abc -genlib <gtech.genlib>
        write_verilog <gtech_file>


    verilog type:
        read_verilog <verilog_file>
        hierarchy -check
        proc; fsm; memory
        techmap
        abc -genlib <gtech.genlib>
        write_verilog <gtech_file>

Step2: Boolean representaiton:
    TODO:


Step3: Synthesis:
    optimizaiton
    technology mapping
    physical design