`timescale 1ns / 1ps

module lif_neuron (
    input  wire               clk,
    input  wire               reset,
    input  wire               enable,
    input  wire signed [31:0] current_in,

    output reg  signed [31:0] membrane_out,
    output reg                spike_out,
    output reg                valid_out
);

    // All currents and membranes have 15 fractional bits.
    // Beta = 29491 / 32768.
    // Threshold = 32768 / 32768 = 1.0.
    localparam signed [47:0] BETA       = 48'sd29491;
    localparam signed [47:0] ROUND_HALF = 48'sd16384;
    localparam signed [47:0] THRESHOLD  = 48'sd32768;

    wire signed [47:0] membrane_extended;
    wire signed [47:0] current_extended;
    wire signed [47:0] beta_product;
    wire signed [47:0] rounded_product;
    wire signed [47:0] leaked_membrane;
    wire signed [47:0] integrated_membrane;
    wire signed [47:0] next_membrane;

    wire previous_reset;
    wire next_spike;

    // Explicit sign extension preserves negative values.
    assign membrane_extended = {
        {16{membrane_out[31]}}, membrane_out
    };

    assign current_extended = {
        {16{current_in[31]}}, current_in
    };

    // Product initially has 30 fractional bits.
    assign beta_product = membrane_extended * BETA;

    // Match Python: (product + 16384) // 32768.
    // >>> performs a signed arithmetic right shift.
    assign rounded_product = beta_product + ROUND_HALF;
    assign leaked_membrane = rounded_product >>> 15;

    assign integrated_membrane =
        leaked_membrane + current_extended;

    // Delayed subtraction reset:
    // determine reset from the PREVIOUS membrane.
    assign previous_reset =
        membrane_extended > THRESHOLD;

    assign next_membrane =
        integrated_membrane
        - (previous_reset ? THRESHOLD : 48'sd0);

    // Equality with threshold does NOT produce a spike.
    assign next_spike = next_membrane > THRESHOLD;

    // Synchronous active-high reset.
    always @(posedge clk) begin
        if (reset) begin
            membrane_out <= 32'sd0;
            spike_out    <= 1'b0;
            valid_out    <= 1'b0;
        end
        else if (enable) begin
            // Validated operating ranges fit within 32 bits.
            membrane_out <= next_membrane[31:0];
            spike_out    <= next_spike;
            valid_out    <= 1'b1;
        end
        else begin
            // Preserve membrane between accepted time steps.
            membrane_out <= membrane_out;
            spike_out    <= 1'b0;
            valid_out    <= 1'b0;
        end
    end

endmodule