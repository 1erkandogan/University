module hazardUnit(
    input [3:0] RA1D,
    input [3:0] RA2D,
    input [3:0] RA1E, 
    input [3:0] RA2E,
    input [3:0] WA3E,
    input [3:0] WA3M,
    input [3:0] WA3W,
    input RegWriteM,
    input RegWriteW,
    input MemtoRegE,
    input BranchE,
    input conditionCheck,
    input PCSrcD,
    input PCSrcE,
    input PCSrcM,
    input PCSrcW,

    output reg [1:0] ForwardAE,
    output reg [1:0] ForwardBE,
    output reg BranchTakenE,
    output reg PCWrPendingF,
    output reg StallF,
    output reg StallD,
    output reg FlushD,
    output reg FlushE,
    output reg LDRStall
);

always @(*)
begin
    if ((RA1E == WA3M) & RegWriteM)
    begin
        ForwardAE = 2'b10;
    end
    else if ((RA1E == WA3W) & RegWriteW)
    begin
        ForwardAE = 2'b01;
    end
    else
    begin
        ForwardAE = 2'b00;
    end


    if ((RA2E == WA3M) & RegWriteM)
    begin
        ForwardBE = 2'b10;
    end
    else if ((RA2E == WA3W) & RegWriteW)
    begin
        ForwardBE = 2'b01;
    end
    else
    begin
        ForwardBE = 2'b00;
    end


    LDRStall = (RA1D == WA3E | RA2D == WA3E) & MemtoRegE;
    BranchTakenE = BranchE & conditionCheck;
    PCWrPendingF = PCSrcD | PCSrcE | PCSrcM;
    StallF = LDRStall | PCWrPendingF;
    StallD = LDRStall;
    FlushD = PCWrPendingF | PCSrcW | BranchTakenE;
    FlushE = LDRStall | BranchTakenE; 


end
endmodule
