module Register_exec
    (
	  input  clk, reset,
	  input PCSrcD, BranchD, RegWriteD, MemWriteD, MemtoRegD, ALUSrcD, FlagWriteD,
	  input [31:0] RD1D, RD2D,
	  input [3:0] WA3D,
	  input [31:0] ExtImmD,
	  input [3:0] CondD,
	  input [3:0] Flags,
	  input [3:0] ALUControlD,
	  
	  output reg PCSrcE, BranchE, RegWriteE, MemWriteE, MemtoRegE, ALUSrcE, FlagWriteE,
	  output reg [31:0] RD1E, RD2E,
	  output reg [3:0] WA3E,
	  output reg [31:0] ExtImmE,
	  output reg [3:0] CondE,
	  output reg [3:0] FlagsE,
	  output reg [3:0] ALUControlE
    );
	 
always@(posedge clk) begin
	if(reset == 1'b0)
		begin
		PCSrcE <= PCSrcD;
		BranchE <= BranchD;
		RegWriteE <= RegWriteD;
		MemWriteE <= MemWriteD;
		MemtoRegE <= MemtoRegD;
		ALUControlE <= ALUControlD;
		ALUSrcE <= ALUSrcD;
		FlagWriteE <= FlagWriteD;
		CondE <= CondD;
		FlagsE <= Flags;
		RD1E <= RD1D;
		RD2E <= RD2D;
		WA3E <= WA3D;
		ExtImmE <= ExtImmD;
		end

	else
		begin
		PCSrcE <= 0;
		BranchE <= 0;
		RegWriteE <= 0;
		MemWriteE <= 0;
		MemtoRegE <= 0;
		ALUControlE <= {4{1'b0}};
		ALUSrcE <= 0;
		FlagWriteE <= 0;
		CondE <= {4{1'b0}};
		FlagsE <= {4{1'b0}};
		RD1E <= {32{1'b0}};
		RD2E <= {32{1'b0}};
		WA3E <= {4{1'b0}};
		ExtImmE <= {32{1'b0}};
		end
end
	 
endmodule
