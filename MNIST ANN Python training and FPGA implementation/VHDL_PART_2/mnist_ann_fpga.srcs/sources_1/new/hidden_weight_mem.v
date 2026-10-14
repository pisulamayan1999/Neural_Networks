module hidden_weight_mem (
    input  wire [9:0] addr,
    output reg signed [15:0] weight_out
);

    reg signed [15:0] weights [0:783];

    initial begin
        $readmemh("hidden_weight0.mem", weights);
    end

    always @(*) begin
        weight_out = weights[addr];
    end

endmodule