`timescale 1ns / 1ps

module snn_network (
    input wire clk,
    input wire reset,

    // Load one hidden neuron's current through the MAC.
    input wire current_start,
    input wire [3:0] neuron_index,
    input wire signed [15:0] bias_in,
    input wire input_valid,
    input wire [15:0] pixel_in,
    input wire signed [15:0] weight_in,

    output wire current_busy,
    output wire current_loaded,
    output wire all_currents_ready,

    // Pulse after all 16 currents have been loaded.
    // Reset is required before processing another image.
    input wire run_start,
    output wire run_busy,
    output wire done,
    output wire [3:0] prediction,
    output wire [39:0] counts_out,

    // Verification outputs.
    output wire hidden_valid,
    output wire [15:0] hidden_spikes,
    output wire [511:0] hidden_membranes,

    output wire output_valid,
    output wire [319:0] output_currents,
    output wire [319:0] output_membranes,
    output wire [9:0] output_spikes
);

    localparam [2:0] IDLE       = 3'd0;
    localparam [2:0] HIDDEN     = 3'd1;
    localparam [2:0] OUTPUT     = 3'd2;
    localparam [2:0] COUNT      = 3'd3;
    localparam [2:0] WAIT_COUNT = 3'd4;
    localparam [2:0] FINISHED   = 3'd5;

    reg [2:0] state;

    wire output_ready;
    wire hidden_step;
    wire output_step;
    wire allow_load;

    assign allow_load = state == IDLE;
    assign hidden_step = state == HIDDEN;

    // The output layer consumes the hidden spikes on
    // the clock following the hidden-layer update.
    assign output_step =
        (state == OUTPUT) && hidden_valid;

    assign run_busy =
        (state != IDLE) && (state != FINISHED);

    snn_hidden_layer hidden_layer (
        .clk                (clk),
        .reset              (reset),
        .current_start      (current_start && allow_load),
        .neuron_index       (neuron_index),
        .bias_in            (bias_in),
        .input_valid        (input_valid && allow_load),
        .pixel_in           (pixel_in),
        .weight_in          (weight_in),
        .current_busy       (current_busy),
        .current_loaded     (current_loaded),
        .all_currents_ready (all_currents_ready),
        .step_enable        (hidden_step),
        .spikes_out         (hidden_spikes),
        .membranes_out      (hidden_membranes),
        .step_valid         (hidden_valid)
    );

    snn_output_layer output_layer (
        .clk           (clk),
        .reset         (reset),
        .step_enable   (output_step),
        .hidden_spikes (hidden_spikes),
        .ready         (output_ready),
        .step_valid    (output_valid),
        .done          (done),
        .currents_out  (output_currents),
        .membranes_out (output_membranes),
        .spikes_out    (output_spikes),
        .counts_out    (counts_out),
        .prediction    (prediction)
    );

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
        end
        else begin
            case (state)
                IDLE: begin
                    if (
                        run_start
                        && all_currents_ready
                        && !current_busy
                        && !current_start
                        && !current_loaded
                    )
                        state <= HIDDEN;
                end

                HIDDEN:
                    state <= OUTPUT;

                OUTPUT:
                    state <= COUNT;

                COUNT:
                    state <= WAIT_COUNT;

                WAIT_COUNT: begin
                    if (done)
                        state <= FINISHED;
                    else
                        state <= HIDDEN;
                end

                FINISHED:
                    state <= FINISHED;

                default:
                    state <= IDLE;
            endcase
        end
    end

endmodule