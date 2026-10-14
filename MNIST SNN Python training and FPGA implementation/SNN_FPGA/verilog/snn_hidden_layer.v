`timescale 1ns / 1ps

module snn_hidden_layer (
    input wire clk,
    input wire reset,

    // Calculate one selected neuron's current.
    // Accept current_start only while current_busy=0.
    input wire current_start,
    input wire [3:0] neuron_index,
    input wire signed [15:0] bias_in,

    input wire input_valid,
    input wire [15:0] pixel_in,
    input wire signed [15:0] weight_in,

    output wire current_busy,
    output reg current_loaded,
    output wire all_currents_ready,

    // One accepted pulse advances all 16 LIF neurons
    // by one SNN time step.
    input wire step_enable,

    output wire [15:0] spikes_out,
    output wire [511:0] membranes_out,
    output wire step_valid
);

    wire mac_done;
    wire signed [47:0] mac_accumulator;
    wire signed [31:0] mac_current;

    reg [3:0] selected_neuron;
    reg [15:0] loaded_mask;
    reg signed [31:0] stored_currents [0:15];

    wire [15:0] neuron_valid;
    wire lif_enable;

    integer index;
    genvar neuron;

    hidden_current_mac mac (
        .clk             (clk),
        .reset           (reset),
        .start           (current_start),
        .bias_in         (bias_in),
        .input_valid     (input_valid),
        .pixel_in        (pixel_in),
        .weight_in       (weight_in),
        .busy            (current_busy),
        .done            (mac_done),
        .accumulator_out (mac_accumulator),
        .current_out     (mac_current)
    );

    assign all_currents_ready = &loaded_mask;

    // Prevent advancing membranes while loading a current.
    assign lif_enable =
        step_enable
        && all_currents_ready
        && !current_busy
        && !current_start
        && !mac_done;

    assign step_valid = &neuron_valid;

    always @(posedge clk) begin
        if (reset) begin
            selected_neuron <= 4'd0;
            loaded_mask     <= 16'd0;
            current_loaded  <= 1'b0;

            for (index = 0; index < 16; index = index + 1)
                stored_currents[index] <= 32'sd0;
        end
        else begin
            current_loaded <= 1'b0;

            if (current_start && !current_busy) begin
                selected_neuron <= neuron_index;
                loaded_mask[neuron_index] <= 1'b0;
            end

            // MAC outputs become available after its completion
            // edge. Capture them on the following clock.
            if (mac_done) begin
                stored_currents[selected_neuron] <= mac_current;
                loaded_mask[selected_neuron] <= 1'b1;
                current_loaded <= 1'b1;
            end
        end
    end

    generate
        for (neuron = 0; neuron < 16; neuron = neuron + 1) begin
            lif_neuron lif (
                .clk          (clk),
                .reset        (reset),
                .enable       (lif_enable),
                .current_in   (stored_currents[neuron]),
                .membrane_out (membranes_out[neuron*32 +: 32]),
                .spike_out    (spikes_out[neuron]),
                .valid_out    (neuron_valid[neuron])
            );
        end
    endgenerate

endmodule