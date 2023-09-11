import cocotb
from cocotb.triggers import Timer
from cocotb.clock import Clock
from cocotb.handle import Force, Release
from cocotb.triggers import (
    First,
    Join,
    NextTimeStep,
    ReadOnly,
    ReadWrite,
    RisingEdge,
    FallingEdge,
    Timer,
    TriggerException,
)

def regFileInit(dut):

    dut.eru446.b2v_regfile.R0.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R1.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R2.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R3.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R4.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R5.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R6.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R7.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R8.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R9.dataOut.value   = 0x00000000
    dut.eru446.b2v_regfile.R10.dataOut.value  = 0x00000000
    dut.eru446.b2v_regfile.R11.dataOut.value  = 0x00000000
    dut.eru446.b2v_regfile.R12.dataOut.value  = 0x00000000
    dut.eru446.b2v_regfile.R13.dataOut.value  = 0x00000000
    dut.eru446.b2v_regfile.R14.dataOut.value  = 0x00000000
    dut.eru446.b2v_regfile.R15.dataOut.value  = 0x00000000

def dumpRegFile(dut):
    cocotb.log.info("RegFileDump")
    try:
        cocotb.log.info("R0  = %s" % hex(dut.eru446.b2v_regfile.R0.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R1  = %s" % hex(dut.eru446.b2v_regfile.R1.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R2  = %s" % hex(dut.eru446.b2v_regfile.R2.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R3  = %s" % hex(dut.eru446.b2v_regfile.R3.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R4  = %s" % hex(dut.eru446.b2v_regfile.R4.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R5  = %s" % hex(dut.eru446.b2v_regfile.R5.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R6  = %s" % hex(dut.eru446.b2v_regfile.R6.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R7  = %s" % hex(dut.eru446.b2v_regfile.R7.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R8  = %s" % hex(dut.eru446.b2v_regfile.R8.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R9  = %s" % hex(dut.eru446.b2v_regfile.R9.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R10 = %s" % hex(dut.eru446.b2v_regfile.R10.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R11 = %s" % hex(dut.eru446.b2v_regfile.R11.dataOut.value))
    except:
        pass    
    try:
        cocotb.log.info("R12 = %s" % hex(dut.eru446.b2v_regfile.R12.dataOut.value))
    except:
        pass
    try:
        cocotb.log.info("R13 = %s" % hex(dut.eru446.b2v_regfile.R13.dataOut.value))
    except:
        pass
    try:
        cocotb.log.info("R14 = %s" % hex(dut.eru446.b2v_regfile.R14.dataOut.value))
    except:
        pass
    try:
        cocotb.log.info("R15 = %s\n\n" % hex(dut.eru446.b2v_regfile.R15.dataOut.value))
    except:
        cocotb.log.info("\n\n")

def dumpDataMemory(dut):
    cocotb.log.info("Dumping Data Memory")
    for i in range(0,128,4):
        try:
            cocotb.log.info(f"Data[{str(i).rjust(3, '0')}] = 0x{hex(dut.eru446.b2v_data_mem.mem[i+3].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_data_mem.mem[i+2].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_data_mem.mem[i+1].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_data_mem.mem[i].value)[2:].rjust(2, '0')}")
        except:
            pass
    cocotb.log.info("\n\n")

def dumpInstMemory(dut):
    cocotb.log.info("Dumping Instruction Memory")
    for i in range(0,128,4):
        try:
            cocotb.log.info(f"Data[{str(i).rjust(3, '0')}] = 0x{hex(dut.eru446.b2v_inst_mem.mem[i+3].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_inst_mem.mem[i+2].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_inst_mem.mem[i+1].value)[2:].rjust(2, '0')}{hex(dut.eru446.b2v_inst_mem.mem[i].value)[2:].rjust(2, '0')}")
        except:
            pass
    cocotb.log.info("\n\n")

def dumpSignals(dut):
    cocotb.log.info("Dumping Signals")
    try:
        cocotb.log.info(f"PC                = {hex(dut.PCF.value)}")
    except:
        cocotb.log.info(f"PC                = {dut.PCF.value}")

    try:
        cocotb.log.info(f"Instruction       = {hex(dut.Instruction.value)}")
    except:
        cocotb.log.info(f"Instruction       = {dut.Instruction.value}")

    try:
        cocotb.log.info(f"RegSrc            = {dut.RegSrc.value}")
    except:
        pass

    try:
        cocotb.log.info(f"RegWriteW         = {dut.RegWriteW.value}")
    except:
        pass

    try:
        cocotb.log.info(f"ALUSrcE           = {dut.ALUSrcE.value}")
    except:
        pass

    try:
        cocotb.log.info(f"MemtoRegW         = {dut.MemtoRegW.value}")
    except:
        pass

    try:
        cocotb.log.info(f"MemWriteM         = {dut.MemWriteM.value}")
    except:
        pass

    try:
        cocotb.log.info(f"Flags (CNZV)      = {dut.FlagC.value}{dut.FlagN.value}{dut.FlagZ.value}{dut.FlagV.value}")
    except:
        pass
        
    try:
        cocotb.log.info(f"ALUOp             = {dut.ALUOp.value}")
    except:
        pass

    try:
        cocotb.log.info(f"ImmSrcD           = {dut.ImmSrcD.value}\n\n")
    except:
        pass

def dumpDatapathRegisters(dut):
    cocotb.log.info("Dumping Datapath Registers")

    try:
        cocotb.log.info(f"Fetch                     = {hex(dut.b2v_datapathDesign.b2v_FetchR.dataOut.value)}\n\n")
    except:
        cocotb.log.info(f"Fetch                     = {dut.b2v_datapathDesign.b2v_FetchR.dataOut.value}\n\n")

    try:
        cocotb.log.info(f"Decode                    = {hex(dut.b2v_datapathDesign.b2v_DecodeR.dataOut.value)}\n\n")
    except:
        cocotb.log.info(f"Decode                    = {dut.b2v_datapathDesign.b2v_DecodeR.dataOut.value}\n\n")

    try:
        cocotb.log.info(f"RA1D                      = {hex(dut.b2v_datapathDesign.b2v_inst5.y.value)}\n\n")
    except:
        cocotb.log.info(f"RA1D                      = {dut.b2v_datapathDesign.b2v_inst5.y.value}\n\n")

    try:
        cocotb.log.info(f"RA2D                      = {hex(dut.b2v_datapathDesign.b2v_inst6.y.value)}\n\n")
    except:
        cocotb.log.info(f"RA2D                      = {dut.b2v_datapathDesign.b2v_inst6.y.value}\n\n")

    try:
        cocotb.log.info(f"RegFile DataOut1          = {hex(dut.eru446.b2v_regfile.dataOut1.value)}\n\n")
    except:
        cocotb.log.info(f"RegFile DataOut1          = {dut.eru446.b2v_regfile.dataOut1.value}\n\n")

    try:
        cocotb.log.info(f"RegFile DataOut2          = {hex(dut.eru446.b2v_regfile.dataOut2.value)}\n\n")
    except:
        cocotb.log.info(f"RegFile DataOut2          = {dut.eru446.b2v_regfile.dataOut2.value}\n\n")

    try:
        cocotb.log.info(f"Execute - dataOut1        = {hex(dut.b2v_datapathDesign.b2v_inst7.dataOut.value)}")
    except:
        cocotb.log.info(f"Execute - dataOut1        = {dut.b2v_datapathDesign.b2v_inst7.dataOut.value}")

    try:
        cocotb.log.info(f"Execute - dataOut2        = {hex(dut.b2v_datapathDesign.b2v_inst9.dataOut.value)}")
    except:
        cocotb.log.info(f"Execute - dataOut2        = {dut.b2v_datapathDesign.b2v_inst9.dataOut.value}")

    try:
        cocotb.log.info(f"Execute - Immediate       = {hex(dut.b2v_datapathDesign.b2v_inst10.dataOut.value)}")
    except:
        cocotb.log.info(f"Execute - Immediate       = {dut.b2v_datapathDesign.b2v_inst10.dataOut.value}")

    try:
        cocotb.log.info(f"Execute - Rd              = {hex(dut.b2v_datapathDesign.b2v_inst11.dataOut.value)}\n\n")
    except:
        cocotb.log.info(f"Execute - Rd              = {dut.b2v_datapathDesign.b2v_inst11.dataOut.value}\n\n")

    try:
        cocotb.log.info(f"Memory - ALUResult        = {hex(dut.b2v_datapathDesign.b2v_inst15.dataOut.value)}")
    except:
        cocotb.log.info(f"Memory - ALUResult        = {dut.b2v_datapathDesign.b2v_inst15.dataOut.value}")

    try:
        cocotb.log.info(f"Memory - WriteData        = {hex(dut.b2v_datapathDesign.b2v_inst16.dataOut.value)}")
    except:
        cocotb.log.info(f"Memory - WriteData        = {dut.b2v_datapathDesign.b2v_inst16.dataOut.value}")

    try:
        cocotb.log.info(f"Memory - Rd               = {hex(dut.b2v_datapathDesign.b2v_inst17.dataOut.value)}\n\n")
    except:
        cocotb.log.info(f"Memory - Rd               = {dut.b2v_datapathDesign.b2v_inst17.dataOut.value}\n\n")

    try:
        cocotb.log.info(f"WriteBack - ReadData      = {hex(dut.b2v_datapathDesign.b2v_inst20.dataOut.value)}")
    except:
        cocotb.log.info(f"WriteBack - ReadData      = {dut.b2v_datapathDesign.b2v_inst20.dataOut.value}")

    try:
        cocotb.log.info(f"WriteBack - ALUResult     = {hex(dut.b2v_datapathDesign.b2v_inst21.dataOut.value)}")
    except:
        cocotb.log.info(f"WriteBack - ALUResult     = {dut.b2v_datapathDesign.b2v_inst21.dataOut.value}")

    try:
        cocotb.log.info(f"WriteBack - Rd            = {hex(dut.b2v_datapathDesign.b2v_inst22.dataOut.value)}\n\n")
    except:
        cocotb.log.info(f"WriteBack - Rd            = {dut.b2v_datapathDesign.b2v_inst22.dataOut.value}\n\n")

def dumpControlUnit(dut):
    cocotb.log.info("Dumping Control Unit")
    cocotb.log.info(f"Stage 1 - Input                   = {dut.b2v_controllerUnit.b2v_controllerStageOne.InstructionD.value}")
    cocotb.log.info(f"ImmSrcD                           = {dut.b2v_controllerUnit.b2v_controllerStageOne.ImmSrcD.value}")
    cocotb.log.info(f"RegSrcD                           = {dut.b2v_controllerUnit.b2v_controllerStageOne.RegSrcD.value}")
    cocotb.log.info(f"Stage 1 - Output                  = {dut.b2v_controllerUnit.b2v_controllerStageOne.InstructionE.value}\n")
    cocotb.log.info(f"Stage 2 - Input                   = {dut.b2v_controllerUnit.b2v_controllerStageTwo.InstructionE.value}")
    cocotb.log.info(f"ALUSrcBe                          = {dut.b2v_controllerUnit.b2v_controllerStageTwo.ALUSrcBE.value}")
    cocotb.log.info(f"ALUOp                             = {dut.b2v_controllerUnit.b2v_controllerStageTwo.ALUOpE.value}")
    cocotb.log.info(f"Stage 2 - Output                  = {dut.b2v_controllerUnit.b2v_controllerStageTwo.InstructionM.value}\n")
    cocotb.log.info(f"Stage 3 - Input                   = {dut.b2v_controllerUnit.b2v_controllerStageThree.InstructionM.value}")
    cocotb.log.info(f"MemWriteM                         = {dut.b2v_controllerUnit.b2v_controllerStageThree.MemWriteM.value}")
    cocotb.log.info(f"Stage 3 - Output                  = {dut.b2v_controllerUnit.b2v_controllerStageThree.InstructionW.value}\n")
    cocotb.log.info(f"Stage 4 - Input                   = {dut.b2v_controllerUnit.b2v_controllerStageFour.InstructionW.value}")
    cocotb.log.info(f"MemtoRegW                         = {dut.b2v_controllerUnit.b2v_controllerStageFour.MemToRegW.value}")
    cocotb.log.info(f"RegWriteW                         = {dut.b2v_controllerUnit.b2v_controllerStageFour.RegWriteW.value}\n")

def dumpEverything(dut):
    cocotb.log.info("Dumping Everything")
    cocotb.log.info("Fetch")
    cocotb.log.info(f"PC                        = {hex(dut.b2v_datapathDesign.b2v_FetchR.dataOut.value)}\n")
    
    cocotb.log.info("Decode")
    cocotb.log.info(f"Instruction               = {hex(dut.b2v_datapathDesign.b2v_DecodeR.dataOut.value)}")
    cocotb.log.info(f"RegSrc                    = {dut.RegSrc.value}")
    cocotb.log.info(f"RA1                       = {dut.b2v_datapathDesign.b2v_inst5.y.value}")
    cocotb.log.info(f"RD1                       = {hex(dut.eru446.b2v_regfile.dataOut1.value)}")
    cocotb.log.info(f"RA2                       = {dut.b2v_datapathDesign.b2v_inst6.y.value}")
    cocotb.log.info(f"RD2                       = {hex(dut.eru446.b2v_regfile.dataOut2.value)}")
    cocotb.log.info(f"WA                        = {dut.b2v_datapathDesign.b2v_inst11.dataIn.value}")
    cocotb.log.info(f"ImmSrcD                   = {dut.ImmSrcD.value}")
    cocotb.log.info(f"Immshamt                  = {dut.Immshamt.value}")
    cocotb.log.info(f"ImmshType                 = {dut.ImmshType.value}")
    cocotb.log.info(f"Imm                       = {hex(dut.b2v_datapathDesign.b2v_inst32.dataOut.value)}\n")

    cocotb.log.info(f"Execute")
    cocotb.log.info(f"Instruction               = {hex(dut.b2v_controllerUnit.b2v_controllerStageTwo.InstructionE.value)}")
    cocotb.log.info(f"ALUSrc                    = {dut.ALUSrcE.value}")
    cocotb.log.info(f"ALUOp                     = {dut.ALUOp.value}")
    cocotb.log.info(f"ALUResult                 = {hex(dut.b2v_datapathDesign.b2v_inst15.dataIn.value)}")
    cocotb.log.info(f"WriteData                 = {dut.b2v_datapathDesign.b2v_inst16.dataIn.value}")
    cocotb.log.info(f"WA                        = {dut.b2v_datapathDesign.b2v_inst17.dataIn.value}\n")

    cocotb.log.info(f"Memory")
    cocotb.log.info(f"Instruction               = {hex(dut.b2v_controllerUnit.b2v_controllerStageThree.InstructionM.value)}")
    cocotb.log.info(f"MemWrite                  = {dut.MemWriteM.value}")
    cocotb.log.info(f"MemAddress                = {hex(dut.eru446.b2v_data_mem.ADDR.value)}")
    cocotb.log.info(f"WData                     = {hex(dut.eru446.b2v_data_mem.WD.value)}")
    try:
        cocotb.log.info(f"RData                     = {hex(dut.b2v_datapathDesign.b2v_inst20.dataOut.value)}")
    except:
        cocotb.log.info(f"RData                     = {dut.b2v_datapathDesign.b2v_inst20.dataOut.value}")
    cocotb.log.info(f"ALUResult                 = {hex(dut.b2v_datapathDesign.b2v_inst21.dataIn.value)}")
    cocotb.log.info(f"WA                        = {dut.b2v_datapathDesign.b2v_inst22.dataIn.value}\n")

    cocotb.log.info(f"WriteBack")
    cocotb.log.info(f"Instruction               = {hex(dut.b2v_controllerUnit.b2v_controllerStageFour.InstructionW.value)}")
    cocotb.log.info(f"MemtoReg                  = {dut.MemtoRegW.value}")
    cocotb.log.info(f"RegWrite                  = {dut.RegWriteW.value}")
    try:
        cocotb.log.info(f"RData                     = {hex(dut.b2v_datapathDesign.b2v_inst20.dataOut.value)}")
    except:
        cocotb.log.info(f"RData                     = {dut.b2v_datapathDesign.b2v_inst20.dataOut.value}")
    cocotb.log.info(f"ALUResult                 = {hex(dut.b2v_datapathDesign.b2v_inst21.dataOut.value)}\n")

def dumpSignals2(dut):
    cocotb.log.info("Dumping Signals")
    try:
        cocotb.log.info(f"PC                = {hex(dut.PCF.value)}\n\n")
    except:
        cocotb.log.info(f"PC                = {dut.PCF.value}\n\n")

    try:
        cocotb.log.info(f"InstructionD      = {hex(dut.InstructionD.value)}")
    except:
        cocotb.log.info(f"InstructionD      = {dut.InstructionD.value}")

    cocotb.log.info(f"RegSrcD           = {dut.RegSrcD.value}")
    cocotb.log.info(f"RA1D              = {dut.RA1D.value}")
    cocotb.log.info(f"RA2D              = {dut.RA2D.value}")
    cocotb.log.info(f"dataOut2shamt     = {dut.dataOut2shamt.value}")
    cocotb.log.info(f"dataOut2shType    = {dut.dataOut2shType.value}")
    cocotb.log.info(f"ImmSrcD           = {dut.ImmSrcD.value}")
    cocotb.log.info(f"Immshamt          = {dut.Immshamt.value}")
    cocotb.log.info(f"ImmshType         = {dut.ImmshType.value}")
    cocotb.log.info(f"Imm               = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_17.value}")
    cocotb.log.info(f"ShiftedImm        = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_2.value}\n\n")
    

    try:
        cocotb.log.info(f"InstructionE      = {hex(dut.b2v_controllerUnit.b2v_inst1.InstructionE.value)}")
    except:
        cocotb.log.info(f"InstructionE      = {dut.b2v_controllerUnit.b2v_inst1.InstructionE.value}")

    cocotb.log.info(f"ForwardAE         = {dut.ForwardAE.value}")
    cocotb.log.info(f"ForwardBE         = {dut.ForwardBE.value}")
    cocotb.log.info(f"ALUSrcBE          = {dut.ALUSrcBE.value}")
    cocotb.log.info(f"ALUOpE            = {dut.ALUOpE.value}")

    try:
        cocotb.log.info(f"ALU-A             = {hex(dut.b2v_datapathDesign.SYNTHESIZED_WIRE_5.value)}")
    except:
        cocotb.log.info(f"ALU-A             = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_5.value}")
    try:
        cocotb.log.info(f"Imm after reg     = {hex(dut.b2v_datapathDesign.b2v_inst10.dataOut.value)}")
    except:
        cocotb.log.info(f"Imm after reg     = {dut.b2v_datapathDesign.b2v_inst10.dataOut.value}")
    try:
        cocotb.log.info(f"ALU-B             = {hex(dut.b2v_datapathDesign.SYNTHESIZED_WIRE_6.value)}")
    except:
        cocotb.log.info(f"ALU-B             = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_6.value}")

    try:
        cocotb.log.info(f"ALUResultE        = {hex(dut.b2v_datapathDesign.ALUResultE.value)}\n\n")
    except:
        cocotb.log.info(f"ALUResultE        = {dut.b2v_datapathDesign.ALUResultE.value}\n\n")

    try:
        cocotb.log.info(f"InstructionM      = {hex(dut.b2v_controllerUnit.b2v_inst2.InstructionM.value)}")
    except:
        cocotb.log.info(f"InstructionM      = {dut.b2v_controllerUnit.b2v_inst2.InstructionM.value}")

    try:
        cocotb.log.info(f"WriteDataM        = {hex(dut.b2v_datapathDesign.SYNTHESIZED_WIRE_11.value)}")
    except:
        cocotb.log.info(f"WriteDataM        = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_11.value}")

    try:
        cocotb.log.info(f"ALUResultM        = {hex(dut.b2v_datapathDesign.ALUResultM.value)}")
    except:
        cocotb.log.info(f"ALUResultM        = {dut.b2v_datapathDesign.ALUResultM.value}")

    try:
        cocotb.log.info(f"ReadDataM         = {hex(dut.b2v_datapathDesign.SYNTHESIZED_WIRE_12.value)}")
    except:
        cocotb.log.info(f"ReadDataM         = {dut.b2v_datapathDesign.SYNTHESIZED_WIRE_12.value}")

    cocotb.log.info(f"MemWriteM         = {dut.MemWriteM.value}\n\n")

    try:
        cocotb.log.info(f"InstructionW      = {hex(dut.b2v_controllerUnit.b2v_inst3.InstructionW.value)}")
    except:
        cocotb.log.info(f"InstructionW      = {dut.b2v_controllerUnit.b2v_inst3.InstructionW.value}")

    cocotb.log.info(f"MemtoRegW         = {dut.MemtoRegW.value}")
    cocotb.log.info(f"RegWriteW         = {dut.RegWriteW.value}\n\n")



@cocotb.test()
async def test_pipelineProcessorv2(dut):
    cocotb.log.info("Starting Test")
    cocotb.fork(Clock(dut.clk, 10, units="ns").start())

    dut.rst.value = Force(1)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = Force(0)
    regFileInit(dut)

    cocotb.log.info("Reset Complete")

    cocotb.log.info("Data Memory before cycles")
    dumpDataMemory(dut)

    cocotb.log.info("Instruction Memory before cycles")
    dumpInstMemory(dut)

    cocotb.log.info("Datapath Registers before cycles")
    #dumpDatapathRegisters(dut)

'''
    for i in range(0, 200):
        cocotb.log.info(f"Cycle {i+1}")
        await RisingEdge(dut.clk)
        dumpSignals2(dut)
        dumpRegFile(dut)

'''
