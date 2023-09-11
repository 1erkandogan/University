module Register_wb
    (
	  input  clk, reset,
	  input PCSrcM, RegWriteM, MemtoRegM,
	  input [31:0] ReadDataM,
	  input [31:0] ALUOutM,
	  input [3:0] WA3M,
	  
	  output reg PCSrcW, RegWriteW, MemtoRegW,
	  output reg [31:0] ReadDataW,
	  output reg [31:0] ALUOutW,
	  output reg [3:0] WA3W
    );

	 
always@(posedge clk) begin
	if(reset == 1'b0)
		begin
		PCSrcW <= PCSrcM;
		RegWriteW <= RegWriteM;
		MemtoRegW <= MemtoRegM;
		ReadDataW <= ReadDataM;
		ALUOutW <= ALUOutM;
		WA3W <= WA3M;
		end

	else
		begin
		PCSrcW <= 0;
		RegWriteW <= 0;
		MemtoRegW <= 0;
		ReadDataW <= {32{1'b0}};
		ALUOutW <= {32{1'b0}};
		WA3W <= {4{1'b0}};
		end
end
	 
endmodule
