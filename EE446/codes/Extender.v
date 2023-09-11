module Extender(
input [23:0] A,
input [1:0] select,
output reg [31:0] Q

);

always @(*)
begin
	case(select)
		2'b00: Q = {24'b0,A[7:0]};
		2'b01: Q = {20'b0,A[11:0]};
		2'b10: Q = {8'b0,A[23:0]};
		2'b11: Q = A;
	endcase
end


endmodule
