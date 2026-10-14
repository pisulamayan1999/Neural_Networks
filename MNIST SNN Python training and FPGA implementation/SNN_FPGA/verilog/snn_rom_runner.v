`timescale 1ns / 1ps

module snn_rom_runner #(
    parameter IMAGE_FILE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/multi_images.mem",
    parameter WEIGHT_FILE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc1_weights.mem",
    parameter BIAS_FILE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc1_biases.mem"
) (
    input wire clk,
    input wire reset,

    // Pulse start while busy=0.
    // image_index must be between 0 and 19.
    input wire start,
    input wire [4:0] image_index,

    output wire busy,
    output reg done,
    output reg [3:0] prediction,
    output reg [39:0] counts_out
);

    localparam [3:0] IDLE       = 4'd0;
    localparam [3:0] CLEAR      = 4'd1;
    localparam [3:0] BIAS_WAIT  = 4'd2;
    localparam [3:0] MAC_START  = 4'd3;
    localparam [3:0] ROM_READ   = 4'd4;
    localparam [3:0] SEND_PAIR  = 4'd5;
    localparam [3:0] WAIT_LOAD  = 4'd6;
    localparam [3:0] RUN_PREP   = 4'd7;
    localparam [3:0] RUN_START  = 4'd8;
    localparam [3:0] WAIT_RUN   = 4'd9;

    reg [3:0] state;
    reg [4:0] selected_image;
    reg [3:0] selected_neuron;
    reg [9:0] selected_pixel;

    // Synchronous read style for the large ROMs.
    // Actual BRAM mapping will be checked after synthesis.
    (* rom_style = "block" *)
    reg [15:0] image_rom [0:15679];

    (* rom_style = "block" *)
    reg signed [15:0] weight_rom [0:12543];

    reg signed [15:0] bias_rom [0:15];

    reg [15:0] pixel_data;
    reg signed [15:0] weight_data;
    reg signed [15:0] bias_data;

    wire network_reset;
    wire mac_start;
    wire pair_valid;
    wire inference_start;

    wire current_busy;
    wire current_loaded;
    wire all_currents_ready;
    wire network_busy;
    wire network_done;
    wire [3:0] network_prediction;
    wire [39:0] network_counts;

    initial begin
        $readmemh(IMAGE_FILE, image_rom);
        $readmemh(WEIGHT_FILE, weight_rom);
        $readmemh(BIAS_FILE, bias_rom);
    end

    // Read ROM data on a clock edge.
    // SEND_PAIR occurs on the following clock, after the
    // registered pixel and weight are available.
    always @(posedge clk) begin
        if (state == ROM_READ) begin
            pixel_data <=
                image_rom[selected_image*784 + selected_pixel];

            weight_data <=
                weight_rom[selected_neuron*784 + selected_pixel];
        end

        // A small synchronous bias ROM.
        bias_data <= bias_rom[selected_neuron];
    end

    assign busy = state != IDLE;

    // CLEAR resets all membranes, counters, and readiness
    // flags before each new image.
    assign network_reset = reset || (state == CLEAR);

    assign mac_start = state == MAC_START;
    assign pair_valid = state == SEND_PAIR;
    assign inference_start = state == RUN_START;

    snn_network network (
        .clk                (clk),
        .reset              (network_reset),
        .current_start      (mac_start),
        .neuron_index       (selected_neuron),
        .bias_in            (bias_data),
        .input_valid        (pair_valid),
        .pixel_in           (pixel_data),
        .weight_in          (weight_data),
        .current_busy       (current_busy),
        .current_loaded     (current_loaded),
        .all_currents_ready (all_currents_ready),
        .run_start          (inference_start),
        .run_busy           (network_busy),
        .done               (network_done),
        .prediction         (network_prediction),
        .counts_out         (network_counts),
        .hidden_valid       (),
        .hidden_spikes      (),
        .hidden_membranes   (),
        .output_valid       (),
        .output_currents    (),
        .output_membranes   (),
        .output_spikes      ()
    );

    always @(posedge clk) begin
        if (reset) begin
            state           <= IDLE;
            selected_image  <= 5'd0;
            selected_neuron <= 4'd0;
            selected_pixel  <= 10'd0;
            done            <= 1'b0;
            prediction      <= 4'd0;
            counts_out      <= 40'd0;
        end
        else begin
            case (state)
                IDLE: begin
                    if (start && image_index < 5'd20) begin
                        selected_image  <= image_index;
                        selected_neuron <= 4'd0;
                        selected_pixel  <= 10'd0;
                        done            <= 1'b0;
                        prediction      <= 4'd0;
                        counts_out      <= 40'd0;
                        state           <= CLEAR;
                    end
                end

                CLEAR:
                    state <= BIAS_WAIT;

                // Give the bias ROM time to supply the
                // currently selected neuron's bias.
                BIAS_WAIT:
                    state <= MAC_START;

                MAC_START:
                    state <= ROM_READ;

                ROM_READ:
                    state <= SEND_PAIR;

                SEND_PAIR: begin
                    if (selected_pixel == 10'd783) begin
                        state <= WAIT_LOAD;
                    end
                    else begin
                        selected_pixel <= selected_pixel + 10'd1;
                        state <= ROM_READ;
                    end
                end

                WAIT_LOAD: begin
                    if (current_loaded) begin
                        if (selected_neuron == 4'd15) begin
                            state <= RUN_PREP;
                        end
                        else begin
                            selected_neuron <= selected_neuron + 4'd1;
                            selected_pixel <= 10'd0;
                            state <= BIAS_WAIT;
                        end
                    end
                end

                RUN_PREP: begin
                    if (all_currents_ready && !current_busy)
                        state <= RUN_START;
                end

                RUN_START:
                    state <= WAIT_RUN;

                WAIT_RUN: begin
                    if (network_done) begin
                        prediction <= network_prediction;
                        counts_out <= network_counts;
                        done       <= 1'b1;
                        state      <= IDLE;
                    end
                end

                default:
                    state <= IDLE;
            endcase
        end
    end

endmodule