module decoderUnit
(
    input [31:0] instruction,
    output reg PCSrcD,
    output reg MemtoReg,
    output reg MemWriteD,
    output reg ALUSrcB,
    output reg ImmSrc,
    output reg RegWriteD,
    output reg flagUpdate,
    output reg carry,
    output reg [3:0] ALUOp,
    output reg [1:0] RegSrc
);

always @(*)
begin
    case (instruction[27:26])
        0: // Data Processing
        begin
            ALUOp = instruction[24:21];
            flagUpdate = instruction[20];
            RegSrc = 0;
            ALUSrcB = 0;
            ImmSrc = 0;
            MemtoReg = 0;
            PCSrcD = 0;
            MemWriteD = 0;
            RegWriteD = ~instruction[20];
        end

        1: // Memory
        begin
            ALUOp = 4;
            flagUpdate = 0;
            RegSrc = 2;
            ALUSrcB = 1;
            ImmSrc = 0;
            MemtoReg = instruction[20];
            PCSrcD = 0;
            MemWriteD = ~instruction[20];
            RegWriteD = instruction[20];
        end

        2: // Branch
        if (instruction[31:28] == 4'b1110)
        begin
            ALUOp = 13;
            flagUpdate = 0;
            RegSrc = 0;
            ALUSrcB = 1;
            ImmSrc = 1;
            MemtoReg = 0;
            PCSrcD = 1;
            RegWriteD = 0;
            MemWriteD = 0;

        end
    endcase
end
endmodule