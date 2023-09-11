# ==============================================================================
# Authors:              Doğu Erkan Arkadaş - Utkucan Doğan
#
# Cocotb Testbench:     For Signed Magnitude Adder/Subtractor
#
# Description:
# ------------------------------------
# Several test-benches as example for EE446
#
# License:
# ==============================================================================


import random
import warnings

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from cocotb.triggers import RisingEdge
from cocotb.triggers import Edge
from cocotb.triggers import Timer
from cocotb.binary import BinaryValue
#from cocotb.regression import TestFactory

@cocotb.test()
async def register_file_test(dut):
    #start the clock
    await cocotb.start(Clock(dut.clk, 10, 'us').start(start_high=False))
    #set clkedge as the falling edge for triggers
    clkedge = FallingEdge(dut.clk)
    await clkedge
    dut.reset.value=1
    await clkedge
    dut.reset.value=0
    dut.Reg_15.value=30
    dut.write_enable.value=0
    #check the first 15 registers
    for i in range(14):     
        dut.Source_select_0.value =i
        dut.Source_select_1.value =i+1
        await Timer(1, units="us")
        assert dut.out_0.value == 0
        assert dut.out_1.value == 0
    #check the 16th register which is an inpur from outside    
    dut.Source_select_0.value =15
    await clkedge
    assert dut.out_0.value == 30
    #Write new data to all registers
    dut.write_enable.value=1
    for i in range(15):
        dut.Destination_select.value = i
        dut.DATA.value=i+1
        await clkedge
    dut.write_enable.value=0
    for i in range(14):
        dut.Source_select_0.value =i
        dut.Source_select_1.value =i+1
        await Timer(1, units="us")
        assert dut.out_0.value == i+1
        assert dut.out_1.value == i+2
    # Reset once more    
    dut.reset.value=1
    await clkedge
    dut.reset.value=0
    dut.write_enable.value=0
    #check the first 15 registers
    for i in range(14):     
        dut.Source_select_0.value =i
        dut.Source_select_1.value =i+1
        await Timer(1, units="us")
        assert dut.out_0.value == 0
        assert dut.out_1.value == 0 
    