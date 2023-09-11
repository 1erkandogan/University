module Register_mem
    (
	  input  clk, reset,
	  input PCSrcE_CondEx, RegWriteE_CondEx, MemWriteE_CondEx, MemtoRegE,
	  input [31:0] ALUResultE,
	  input [31:0] WriteDataE,
	  input [3:0] WA3E,
	  
	  output reg PCSrcM, RegWriteM, MemWriteM, MemtoRegM,
	  output reg [31:0] ALUResultM,
	  output reg [31:0] WriteDataM,
	  output reg [3:0] WA3M
	 
    );
	 
always@(posedge clk) begin
	if(reset == 1'b0)
		begin
		PCSrcM <= PCSrcE_CondEx;
		RegWriteM <= RegWriteE_CondEx;
		MemWriteM <= MemWriteE_CondEx;
		MemtoRegM <= MemtoRegE;
		ALUResultM <= ALUResultE;
		WriteDataM <= WriteDataE;
		WA3M <= WA3E;
		end

	else
		begin
		PCSrcM <= 0;
		RegWriteM <= 0;
		MemWriteM <= 0;
		MemtoRegM <= 0;
		ALUResultM <= {32{1'b0}};
		WriteDataM <= {32{1'b0}};
		WA3M <= {3{1'b0}};
		end
end
	 
endmodule
