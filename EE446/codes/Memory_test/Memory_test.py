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
async def Memory_test(dut):
    #start the clock
    await cocotb.start(Clock(dut.clk, 10, 'us').start(start_high=False))
    #set clkedge as the falling edge for triggers
    clkedge = FallingEdge(dut.clk)
    await clkedge
    dut.ADDR.value=4
    dut.WE.value=1
    dut.WD.value = int.from_bytes(bytes.fromhex('ABCD 1111'),"big")
    await clkedge
    dut.ADDR.value=0
    await Timer(1, units="us")
    #First check if the initial value is loaded
    assert dut.RD.value.buff ==bytes.fromhex('0403 0201')
    dut.ADDR.value=4
    await Timer(1, units="us")
    assert dut.RD.value.buff == bytes.fromhex('ABCD 1111')
    dut.ADDR.value=0
    dut.WD.value = int.from_bytes(bytes.fromhex('1111 1111'),"big")
    await clkedge
    await Timer(1, units="us")
    assert dut.RD.value.buff == bytes.fromhex('1111 1111')