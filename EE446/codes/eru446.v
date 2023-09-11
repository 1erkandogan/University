// Copyright (C) 2018  Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions 
// and other software and tools, and its AMPP partner logic 
// functions, and any output files from any of the foregoing 
// (including device programming or simulation files), and any 
// associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License 
// Subscription Agreement, the Intel Quartus Prime License Agreement,
// the Intel FPGA IP License Agreement, or other applicable license
// agreement, including, without limitation, that your use is for
// the sole purpose of programming logic devices manufactured by
// Intel and sold by Intel or its authorized distributors.  Please
// refer to the applicable agreement for further details.

// PROGRAM		"Quartus Prime"
// VERSION		"Version 18.0.0 Build 614 04/24/2018 SJ Lite Edition"
// CREATED		"Thu Jun 15 01:33:11 2023"

module eru446(
	StallF,
	StallD,
	FlushD,
	FlushE,
	clk,
	ForwardAE,
	ForwardBE,
	inp_test,
	PCSrcW,
	InstF,
	test_out
);


input wire	StallF;
input wire	StallD;
input wire	FlushD;
input wire	FlushE;
input wire	clk;
input wire	[1:0] ForwardAE;
input wire	[1:0] ForwardBE;
input wire	[3:0] inp_test;
output wire	PCSrcW;
output wire	[31:0] InstF;
output wire	[3:0] test_out;

wire	[3:0] ALUControlD;
wire	[3:0] ALUControlE;
wire	[31:0] ALUOutM;
wire	[31:0] ALUResultE;
wire	ALUSrcD;
wire	ALUSrcE;
wire	BranchD;
wire	BranchE;
wire	BranchTakenE;
wire	[3:0] CondE;
wire	CondEx;
wire	[3:0] Const15;
wire	[31:0] ExtImmD;
wire	[31:0] ExtImmD_in;
wire	[31:0] ExtImmE;
wire	FlagC;
wire	FlagN;
wire	[3:0] Flags;
wire	[3:0] FlagsE;
wire	FlagV;
wire	FlagWriteD;
wire	FlagWriteE;
wire	FlagZ;
wire	[1:0] ImmSrcD;
wire	[31:0] InstD;
wire	[31:0] InstF_ALTERA_SYNTHESIZED;
wire	MemtoRegD;
wire	MemtoRegE;
wire	MemtoRegM;
wire	MemtoRegW;
wire	MemWriteD;
wire	MemWriteE;
wire	MemWriteTakenE;
wire	[31:0] PCPlus4;
wire	PCSrcD;
wire	PCSrcE;
wire	PCSrcM;
wire	PCSrcTakenE;
wire	PCSrcW_ALTERA_SYNTHESIZED;
wire	[31:0] RD1D;
wire	[31:0] RD1E;
wire	[31:0] RD2D;
wire	[31:0] RD2D_in;
wire	[31:0] RD2E;
wire	[31:0] ReadDataM;
wire	[1:0] RegSrcD;
wire	RegWriteD;
wire	RegWriteE;
wire	RegWriteM;
wire	RegWriteTakenE;
wire	RegWriteW;
wire	[31:0] ResultW;
wire	rst;
wire	[4:0] shamtImm;
wire	[4:0] shamtReg;
wire	[1:0] shImm;
wire	[1:0] shReg;
wire	[31:0] SrcAE;
wire	[31:0] SrcBE;
wire	[3:0] WA3E;
wire	[3:0] WA3M;
wire	[3:0] WA3W;
wire	[31:0] WriteDataM;
wire	[31:0] SYNTHESIZED_WIRE_17;
wire	[31:0] SYNTHESIZED_WIRE_1;
wire	SYNTHESIZED_WIRE_2;
wire	SYNTHESIZED_WIRE_3;
wire	SYNTHESIZED_WIRE_4;
wire	SYNTHESIZED_WIRE_5;
wire	[31:0] SYNTHESIZED_WIRE_6;
wire	[31:0] SYNTHESIZED_WIRE_8;
wire	[0:31] SYNTHESIZED_WIRE_9;
wire	[0:31] SYNTHESIZED_WIRE_10;
wire	[31:0] SYNTHESIZED_WIRE_11;
wire	[31:0] SYNTHESIZED_WIRE_12;
wire	[31:0] SYNTHESIZED_WIRE_13;
wire	SYNTHESIZED_WIRE_14;
wire	[3:0] SYNTHESIZED_WIRE_15;
wire	[3:0] SYNTHESIZED_WIRE_16;

assign	SYNTHESIZED_WIRE_2 = 0;
assign	SYNTHESIZED_WIRE_9 = 0;
assign	SYNTHESIZED_WIRE_10 = 0;




Adder	b2v_adder4(
	.DATA_A(SYNTHESIZED_WIRE_17),
	.DATA_B(SYNTHESIZED_WIRE_1),
	.OUT(PCPlus4));
	defparam	b2v_adder4.WIDTH = 32;


ALU	b2v_ALU(
	.CI(SYNTHESIZED_WIRE_2),
	.control(ALUControlE),
	.DATA_A(SrcAE),
	.DATA_B(SrcBE),
	.CO(FlagC),
	.OVF(FlagV),
	.N(FlagN),
	.Z(FlagZ),
	.OUT(ALUResultE));
	defparam	b2v_ALU.WIDTH = 32;

assign	PCSrcTakenE = CondEx & PCSrcE;

assign	BranchTakenE = CondEx & BranchE;

assign	RegWriteTakenE = CondEx & RegWriteE;

assign	MemWriteTakenE = CondEx & MemWriteE;


conditioncheck	b2v_condchecker(
	.FlagWriteE(FlagWriteE),
	.FlagC(FlagC),
	.FlagN(FlagN),
	.FlagZ(FlagZ),
	.FlagV(FlagV),
	.CondE(CondE),
	.FlagsE(FlagsE),
	.condex(CondEx),
	.Flags(Flags));


constantValueGenerator	b2v_const15gen(
	.y(Const15));
	defparam	b2v_const15gen.value = 15;
	defparam	b2v_const15gen.W = 4;


constantValueGenerator	b2v_const4(
	.y(SYNTHESIZED_WIRE_1));
	defparam	b2v_const4.value = 4;
	defparam	b2v_const4.W = 32;


Controller_v2	b2v_controller2(
	.reset(rst),
	.Funct(InstD[25:20]),
	.Op(InstD[27:26]),
	.Rd(InstD[15:12]),
	.Src2(InstD[11:0]),
	.PCSrcD(PCSrcD),
	.BranchD(BranchD),
	.RegWriteD(RegWriteD),
	.MemWriteD(MemWriteD),
	.MemtoRegD(MemtoRegD),
	.ALUSrcD(ALUSrcD),
	.FlagWriteD(FlagWriteD),
	.ALUControlD(ALUControlD),
	.ImmSrcD(ImmSrcD),
	.RegSrcD(RegSrcD),
	.shamtImm(shamtImm),
	.shamtReg(shamtReg),
	.shImm(shImm),
	.shReg(shReg));


Memory	b2v_data_mem(
	.clk(clk),
	.WE(SYNTHESIZED_WIRE_3),
	.ADDR(ALUOutM),
	.WD(WriteDataM),
	.RD(ReadDataM));
	defparam	b2v_data_mem.ADDR_WIDTH = 32;
	defparam	b2v_data_mem.BYTE_SIZE = 4;


Register_sync_rw	b2v_decode_reg(
	.clk(clk),
	.reset(FlushD),
	.we(SYNTHESIZED_WIRE_4),
	.DATA(InstF_ALTERA_SYNTHESIZED),
	.OUT(InstD));
	defparam	b2v_decode_reg.WIDTH = 32;

assign	SYNTHESIZED_WIRE_5 =  ~StallF;

assign	SYNTHESIZED_WIRE_4 =  ~StallD;


Extender	b2v_extender(
	.A(InstD[23:0]),
	.select(ImmSrcD),
	.Q(ExtImmD_in));


Register_sync_rw	b2v_fetch_reg(
	.clk(clk),
	.reset(rst),
	.we(SYNTHESIZED_WIRE_5),
	.DATA(SYNTHESIZED_WIRE_6),
	.OUT(SYNTHESIZED_WIRE_17));
	defparam	b2v_fetch_reg.WIDTH = 32;




shifter	b2v_imm_shifter(
	.control(shImm),
	.DATA(ExtImmD_in),
	.shamt(shamtImm),
	.OUT(ExtImmD));
	defparam	b2v_imm_shifter.WIDTH = 32;




Instruction_memory	b2v_inst_mem(
	.ADDR(SYNTHESIZED_WIRE_17),
	.RD(InstF_ALTERA_SYNTHESIZED));
	defparam	b2v_inst_mem.ADDR_WIDTH = 32;
	defparam	b2v_inst_mem.BYTE_SIZE = 4;


Mux_2to1	b2v_mux1(
	.select(PCSrcW_ALTERA_SYNTHESIZED),
	.input_0(PCPlus4),
	.input_1(ResultW),
	.output_value(SYNTHESIZED_WIRE_8));
	defparam	b2v_mux1.WIDTH = 32;


Mux_2to1	b2v_mux2(
	.select(BranchTakenE),
	.input_0(SYNTHESIZED_WIRE_8),
	.input_1(ALUResultE),
	.output_value(SYNTHESIZED_WIRE_6));
	defparam	b2v_mux2.WIDTH = 32;


Mux_2to1	b2v_mux3(
	.select(RegSrcD[0]),
	.input_0(InstD[19:16]),
	.input_1(Const15),
	.output_value(SYNTHESIZED_WIRE_15));
	defparam	b2v_mux3.WIDTH = 4;


Mux_2to1	b2v_mux4(
	.select(RegSrcD[1]),
	.input_0(InstD[3:0]),
	.input_1(InstD[15:12]),
	.output_value(SYNTHESIZED_WIRE_16));
	defparam	b2v_mux4.WIDTH = 4;


Mux_4to1	b2v_mux5(
	.input_0(RD1E),
	.input_1(ResultW),
	.input_2(ALUOutM),
	.input_3(SYNTHESIZED_WIRE_9),
	.select(ForwardAE),
	.output_value(SrcAE));
	defparam	b2v_mux5.WIDTH = 32;


Mux_4to1	b2v_mux6(
	.input_0(RD2E),
	.input_1(ResultW),
	.input_2(ALUOutM),
	.input_3(SYNTHESIZED_WIRE_10),
	.select(ForwardBE),
	.output_value(SYNTHESIZED_WIRE_11));
	defparam	b2v_mux6.WIDTH = 32;


Mux_2to1	b2v_mux7(
	.select(ALUSrcE),
	.input_0(SYNTHESIZED_WIRE_11),
	.input_1(ExtImmE),
	.output_value(SrcBE));
	defparam	b2v_mux7.WIDTH = 32;


Mux_2to1	b2v_mux8(
	.select(MemtoRegW),
	.input_0(SYNTHESIZED_WIRE_12),
	.input_1(SYNTHESIZED_WIRE_13),
	.output_value(ResultW));
	defparam	b2v_mux8.WIDTH = 32;

assign	SYNTHESIZED_WIRE_14 =  ~clk;


Register_exec	b2v_reg_exec(
	.clk(clk),
	.reset(FlushE),
	.PCSrcD(PCSrcD),
	.BranchD(BranchD),
	.RegWriteD(RegWriteD),
	.MemWriteD(MemWriteD),
	.MemtoRegD(MemtoRegD),
	.ALUSrcD(ALUSrcD),
	.FlagWriteD(FlagWriteD),
	.ALUControlD(ALUControlD),
	.CondD(InstD[31:28]),
	.ExtImmD(ExtImmD),
	.Flags(Flags),
	.RD1D(RD1D),
	.RD2D(RD2D),
	.WA3D(InstD[15:12]),
	.PCSrcE(PCSrcE),
	.BranchE(BranchE),
	.RegWriteE(RegWriteE),
	.MemWriteE(MemWriteE),
	.MemtoRegE(MemtoRegE),
	.ALUSrcE(ALUSrcE),
	.FlagWriteE(FlagWriteE),
	.ALUControlE(ALUControlE),
	.CondE(CondE),
	.ExtImmE(ExtImmE),
	.FlagsE(FlagsE),
	.RD1E(RD1E),
	.RD2E(RD2E),
	.WA3E(WA3E));


Register_mem	b2v_reg_mem(
	.clk(clk),
	.reset(rst),
	.PCSrcE_CondEx(PCSrcTakenE),
	.RegWriteE_CondEx(RegWriteTakenE),
	.MemWriteE_CondEx(MemWriteTakenE),
	.MemtoRegE(MemtoRegE),
	.ALUResultE(ALUResultE),
	.WA3E(WA3E),
	.WriteDataE(RD2E),
	.PCSrcM(PCSrcM),
	.RegWriteM(RegWriteM),
	.MemWriteM(SYNTHESIZED_WIRE_3),
	.MemtoRegM(MemtoRegM),
	.ALUResultM(ALUOutM),
	.WA3M(WA3M),
	.WriteDataM(WriteDataM));


shifter	b2v_reg_shifter(
	.control(shReg),
	.DATA(RD2D_in),
	.shamt(shamtReg),
	.OUT(RD2D));
	defparam	b2v_reg_shifter.WIDTH = 32;


Register_wb	b2v_reg_wb(
	.clk(clk),
	.reset(rst),
	.PCSrcM(PCSrcM),
	.RegWriteM(RegWriteM),
	.MemtoRegM(MemtoRegM),
	.ALUOutM(ALUOutM),
	.ReadDataM(ReadDataM),
	.WA3M(WA3M),
	.PCSrcW(PCSrcW_ALTERA_SYNTHESIZED),
	.RegWriteW(RegWriteW),
	.MemtoRegW(MemtoRegW),
	.ALUOutW(SYNTHESIZED_WIRE_12),
	.ReadDataW(SYNTHESIZED_WIRE_13),
	.WA3W(WA3W));


Register_file	b2v_regfile(
	.clk(SYNTHESIZED_WIRE_14),
	.write_enable(RegWriteW),
	.reset(rst),
	.DATA(ResultW),
	.Destination_select(WA3W),
	.Reg_15(PCPlus4),
	.Source_select_0(SYNTHESIZED_WIRE_15),
	.Source_select_1(SYNTHESIZED_WIRE_16),
	.out_0(RD1D),
	.out_1(RD2D_in));
	defparam	b2v_regfile.WIDTH = 32;


Register_simple	b2v_test_reg(
	.clk(clk),
	.reset(rst),
	.DATA(inp_test),
	.OUT(test_out));
	defparam	b2v_test_reg.WIDTH = 4;

assign	PCSrcW = PCSrcW_ALTERA_SYNTHESIZED;
assign	InstF = InstF_ALTERA_SYNTHESIZED;
assign	rst = 0;

endmodule
