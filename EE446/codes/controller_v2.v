	module Controller_v2
	(
	input reset,

	input [1:0] Op,
	input [5:0] Funct,
	input [3:0] Rd,
	input [11:0] Src2,
	
	output reg PCSrcD, BranchD, RegWriteD, MemWriteD, MemtoRegD, ALUSrcD, FlagWriteD,
	output reg [1:0] ImmSrcD,
	output reg [1:0] RegSrcD,
	output reg [3:0] ALUControlD,
	output reg [4:0] shamtReg,
	output reg [1:0] shReg,
	output reg [4:0] shamtImm,
	output reg [1:0] shImm
	);
	
	
always @(*) begin
	case (Op)
	2'b00: begin // Data Processing
		PCSrcD = 0;
		BranchD = 0;
		RegWriteD = 1;
		MemWriteD = 0;
		MemtoRegD = 0;
		ALUControlD = Funct[4:1];
		ALUSrcD = Funct[5];
		FlagWriteD = Funct[0];
		ImmSrcD = 2'b00;
		RegSrcD = 2'b00;
		
		if ((~Funct[5]) & (~Src2[4])) begin
			shamtReg = Src2[11:7];
			shReg = Src2[6:5];
			shamtImm = {5'b00000};
			shImm = 2'b00;
		end
		
		else if ((Funct[5])) begin
			shamtReg = {5'b00000};
			shReg = 2'b00;
			shamtImm = {1'b0, Src2[11:8]};
			shImm = 2'b11;
		end
		
		else begin
			shamtReg = {5'b00000};
			shReg = 2'b00;
			shamtImm = {5'b00000};
			shImm = 2'b00;
		end
	end
	
	2'b01: begin // Memory
		PCSrcD = 0;
		BranchD = 0;
		RegWriteD = Funct[0];
		MemWriteD = ~Funct[0];
		MemtoRegD = 1;
		ALUControlD = 4'b0100;
		ALUSrcD = 1;
		FlagWriteD = 0;
		ImmSrcD = 2'b01;
		RegSrcD = 2'b10;
	end
	
	2'b10: begin // Branch
		PCSrcD = 1;
		BranchD = 1;
		RegWriteD = 0;
		MemWriteD = 0;
		MemtoRegD = 0;
		ALUControlD = 4'b0100;
		ALUSrcD = 1;
		FlagWriteD = 0;
		ImmSrcD = 2'b10;
		RegSrcD = 2'b01;
		
		shamtReg = {5'b00000};
		shReg = 2'b00;
		shamtImm = {5'b00010};
		shImm = 2'b00;
	end
	endcase	
end

endmodule
	