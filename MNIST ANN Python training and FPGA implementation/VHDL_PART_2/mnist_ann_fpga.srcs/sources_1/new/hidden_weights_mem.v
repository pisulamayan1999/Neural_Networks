module hidden_weights_mem (
    input  wire [14:0] addr,
    output reg signed [15:0] weight_out
);

    reg signed [15:0] weights [0:25087];

    initial begin
        $readmemh("hidden_weights.mem", weights);
    end

    always @(*) begin
        weight_out = weights[addr];
    end

endmodule