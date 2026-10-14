`timescale 1ns / 1ps

module hidden_current_mac (
    input  wire               clk,
    input  wire               reset,

    // Pulse start while busy=0.
    // bias_in is sampled on that clock edge.
    input  wire               start,
    input  wire signed [15:0] bias_in,

    // Each clock with busy=1 and input_valid=1
    // accepts one pixel/weight pair.
    input  wire               input_valid,
    input  wire        [15:0] pixel_in,
    input  wire signed [15:0] weight_in,

    output reg                busy,
    output reg                done,

    output reg  signed [47:0] accumulator_out,
    output reg  signed [31:0] current_out
);

    localparam signed [47:0] ROUND_HALF = 48'sd16384;

    reg [9:0] pair_count;
    reg signed [47:0] accumulator;

    wire signed [16:0] pixel_signed;
    wire signed [16:0] weight_signed;
    wire signed [33:0] product;
    wire signed [47:0] product_extended;
    wire signed [47:0] bias_extended;
    wire signed [47:0] next_accumulator;
    wire signed [47:0] rounded_accumulator;
    wire signed [47:0] scaled_current;

    // Pixels are unsigned: 32768 means +1.0.
    // Add a zero sign bit before signed multiplication.
    assign pixel_signed = {1'b0, pixel_in};

    // Weights are signed. Extend their sign bit.
    assign weight_signed = {weight_in[15], weight_in};

    // Pixel and weight each have 15 fractional bits.
    // Product therefore has 30 fractional bits.
    assign product = pixel_signed * weight_signed;

    assign product_extended = {
        {14{product[33]}}, product
    };

    assign bias_extended = {
        {32{bias_in[15]}}, bias_in
    };

    assign next_accumulator =
        accumulator + product_extended;

    // Same final rounding as the validated Python model.
    assign rounded_accumulator =
        next_accumulator + ROUND_HALF;

    assign scaled_current =
        rounded_accumulator >>> 15;

    always @(posedge clk) begin
        if (reset) begin
            pair_count      <= 10'd0;
            accumulator     <= 48'sd0;
            accumulator_out <= 48'sd0;
            current_out     <= 32'sd0;
            busy            <= 1'b0;
            done            <= 1'b0;
        end
        else begin
            // done is a one-clock completion pulse.
            done <= 1'b0;

            if (!busy) begin
                if (start) begin
                    // Align Q15 bias to the Q30 product scale.
                    accumulator <= bias_extended <<< 15;
                    pair_count  <= 10'd0;
                    busy        <= 1'b1;
                end
            end
            else if (input_valid) begin
                accumulator <= next_accumulator;

                if (pair_count == 10'd783) begin
                    // Include the final product in both outputs.
                    accumulator_out <= next_accumulator;
                    current_out     <= scaled_current[31:0];
                    busy            <= 1'b0;
                    done            <= 1'b1;
                end
                else begin
                    pair_count <= pair_count + 10'd1;
                end
            end
        end
    end

endmodule