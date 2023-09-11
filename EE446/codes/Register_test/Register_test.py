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
async def register_test(dut):
    #start the clock
    await cocotb.start(Clock(dut.clk, 10, 'us').start(start_high=False))
    #set clkedge as the falling edge for triggers
    clkedge = FallingEdge(dut.clk)
    await clkedge
    dut.reset.value=1
    await clkedge
    #check if the module behaves correctly
    assert dut.OUT.value == 0
    dut.DATA.value=10
    dut.we.value=1
    dut.reset.value=0
    await clkedge
    assert dut.OUT.value == 10
    dut.reset.value=1
    await clkedge
    #check if the module behaves correctly
    assert dut.OUT.value == 0
    