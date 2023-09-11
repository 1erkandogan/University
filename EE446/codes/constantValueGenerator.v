module constantValueGenerator
#(parameter W = 32, value = 4)
(
	output wire [W-1:0] y
);

assign y = value;

endmodule