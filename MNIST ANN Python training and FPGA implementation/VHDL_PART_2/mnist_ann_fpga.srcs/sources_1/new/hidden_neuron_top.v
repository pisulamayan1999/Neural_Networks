module hidden_neuron_top (
    input  wire clk,
    input  wire reset,
    input  wire start,

    input  wire signed [15:0] pixel_in,

    output wire [9:0] pixel_index,
    output wire done,
    output wire signed [15:0] neuron_out
);

    wire signed [15:0] weight_value;

    // Neuron 0 bias from hidden_bias_q15.txt
    wire signed [15:0] bias_value = 16'sd5339;

    hidden_weight_mem weight_memory (
        .addr(pixel_index),
        .weight_out(weight_value)
    );

    hidden_neuron neuron (
        .clk(clk),
        .reset(reset),
        .start(start),

        .pixel_in(pixel_in),
        .weight_in(weight_value),
        .bias_in(bias_value),

        .pixel_index(pixel_index),
        .done(done),
        .neuron_out(neuron_out)
    );

endmodule