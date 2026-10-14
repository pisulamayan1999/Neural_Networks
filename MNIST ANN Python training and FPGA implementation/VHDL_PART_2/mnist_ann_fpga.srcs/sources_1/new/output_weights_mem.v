module output_weights_mem (
    input  wire [8:0] addr,
    output reg signed [15:0] weight_out
);

    // 10 output neurons x 32 hidden outputs = 320 weights
    reg signed [15:0] weights [0:319];

    initial begin
        $readmemh("output_weights.mem", weights);
    end

    always @(*) begin
        weight_out = weights[addr];
    end

endmodule