module conditioncheck
(
	input FlagWriteE,
   input FlagC, FlagN, FlagZ, FlagV,
	input [3:0] CondE,
	input [3:0] FlagsE,
	output reg condex,
	output reg [3:0] Flags
);

initial
begin
	Flags = 4'b0000;
end


always @(*)

begin
	if (FlagWriteE)
	begin
		Flags = {FlagN, FlagZ, FlagC, FlagV};
	end
	
	else
	begin
	Flags = FlagsE;
	end
	case (CondE)
		4'b0000: condex = FlagZ;
		4'b0001: condex = ~FlagZ;
		4'b0010: condex = FlagC;
		4'b0011: condex = ~FlagC;
		4'b0100: condex = FlagN;
		4'b0101: condex = ~FlagN;
		4'b0110: condex = FlagV;
		4'b0111: condex = ~FlagV;
		4'b1000: condex = FlagC & ~FlagZ;
		4'b1001: condex = ~FlagC | FlagZ;
		4'b1010: condex = FlagN == FlagV;
		4'b1011: condex = FlagN != FlagV;
		4'b1100: condex = ~FlagZ & (FlagN == FlagV);
		4'b1101: condex = FlagZ | (FlagN != FlagV);
		4'b1110: condex = 1;
		4'b1111: condex = 0;
	endcase        

end
endmodule
