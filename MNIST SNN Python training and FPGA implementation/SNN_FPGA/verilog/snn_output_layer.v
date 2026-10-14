`timescale 1ns / 1ps

module snn_output_layer #(
    parameter WEIGHTS_FILE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc2_weights.mem",
    parameter BIASES_FILE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc2_biases.mem"
) (
    input wire clk,
    input wire reset,

    // An input is accepted on a rising edge when
    // step_enable=1 and ready=1.
    input wire step_enable,
    input wire [15:0] hidden_spikes,

    output wire ready,
    output wire step_valid,
    output reg done,

    output reg [319:0] currents_out,
    output wire [319:0] membranes_out,
    output wire [9:0] spikes_out,

    // Four bits per digit: digit 0 in bits [3:0].
    output wire [39:0] counts_out,
    output reg [3:0] prediction
);

    reg signed [15:0] weights [0:159];
    reg signed [15:0] biases [0:9];

    reg signed [31:0] calculated_currents [0:9];
    reg [3:0] spike_counts [0:9];
    reg [3:0] completed_steps;

    wire [9:0] neuron_valid;
    wire accept_step;

    reg [3:0] best_count;

    integer output_index;
    integer hidden_index;
    integer count_index;
    integer prediction_index;
    genvar neuron;

    initial begin
        $readmemh(WEIGHTS_FILE, weights);
        $readmemh(BIASES_FILE, biases);
    end

    assign step_valid = &neuron_valid;

    // Leave one clock after each accepted step for counting
    // the newly generated output spikes.
    assign ready =
        !reset
        && !done
        && !step_valid
        && (completed_steps < 4'd10);

    assign accept_step = step_enable && ready;

    // Spikes are binary, so these sums remain in Q15.
    // Sign-extend every bias and weight before addition.
    always @* begin
        for (
            output_index = 0;
            output_index < 10;
            output_index = output_index + 1
        ) begin
            calculated_currents[output_index] = {
                {16{biases[output_index][15]}},
                biases[output_index]
            };

            for (
                hidden_index = 0;
                hidden_index < 16;
                hidden_index = hidden_index + 1
            ) begin
                if (hidden_spikes[hidden_index]) begin
                    calculated_currents[output_index] =
                        calculated_currents[output_index]
                        + $signed({
                            {16{weights[output_index*16 + hidden_index][15]}},
                            weights[output_index*16 + hidden_index]
                        });
                end
            end
        end
    end

    generate
        for (neuron = 0; neuron < 10; neuron = neuron + 1) begin
            lif_neuron lif (
                .clk          (clk),
                .reset        (reset),
                .enable       (accept_step),
                .current_in   (calculated_currents[neuron]),
                .membrane_out (membranes_out[neuron*32 +: 32]),
                .spike_out    (spikes_out[neuron]),
                .valid_out    (neuron_valid[neuron])
            );

            assign counts_out[neuron*4 +: 4] =
                spike_counts[neuron];
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            currents_out   <= 320'd0;
            completed_steps <= 4'd0;
            done           <= 1'b0;

            for (
                count_index = 0;
                count_index < 10;
                count_index = count_index + 1
            )
                spike_counts[count_index] <= 4'd0;
        end
        else begin
            if (accept_step) begin
                for (
                    count_index = 0;
                    count_index < 10;
                    count_index = count_index + 1
                )
                    currents_out[count_index*32 +: 32] <=
                        calculated_currents[count_index];
            end

            // Count on the clock following the LIF update,
            // when its registered spike and valid are available.
            if (step_valid) begin
                for (
                    count_index = 0;
                    count_index < 10;
                    count_index = count_index + 1
                )
                    spike_counts[count_index] <=
                        spike_counts[count_index]
                        + {3'b000, spikes_out[count_index]};

                completed_steps <= completed_steps + 4'd1;

                if (completed_steps == 4'd9)
                    done <= 1'b1;
            end
        end
    end

    // Strictly greater preserves the lowest digit index
    // when multiple digits have the same count.
    always @* begin
        prediction = 4'd0;
        best_count = spike_counts[0];

        for (
            prediction_index = 1;
            prediction_index < 10;
            prediction_index = prediction_index + 1
        ) begin
            if (spike_counts[prediction_index] > best_count) begin
                best_count = spike_counts[prediction_index];
                prediction = prediction_index;
            end
        end
    end

endmodule