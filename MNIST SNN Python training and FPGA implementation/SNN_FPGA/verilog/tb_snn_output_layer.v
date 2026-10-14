`timescale 1ns / 1ps

module tb_snn_output_layer;

    reg clk;
    reg reset;
    reg step_enable;
    reg [15:0] hidden_spikes;

    wire ready;
    wire step_valid;
    wire done;
    wire [319:0] currents_out;
    wire [319:0] membranes_out;
    wire [9:0] spikes_out;
    wire [39:0] counts_out;
    wire [3:0] prediction;

    reg [7:0] hidden_reference [0:159];
    reg signed [31:0] current_reference [0:99];
    reg signed [31:0] membrane_reference [0:99];
    reg [7:0] spike_reference [0:99];
    reg [7:0] count_reference [0:9];
    reg [7:0] prediction_reference [0:0];

    reg [319:0] held_membranes;
    reg [39:0] held_counts;

    integer running_counts [0:9];

    integer index;
    integer step;
    integer neuron;
    integer address;
    integer checks;
    integer failures;

    snn_output_layer dut (
        .clk           (clk),
        .reset         (reset),
        .step_enable   (step_enable),
        .hidden_spikes (hidden_spikes),
        .ready         (ready),
        .step_valid    (step_valid),
        .done          (done),
        .currents_out  (currents_out),
        .membranes_out (membranes_out),
        .spikes_out    (spikes_out),
        .counts_out    (counts_out),
        .prediction    (prediction)
    );

    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    initial begin
        #100000;
        $fatal(1, "Output-layer test timed out.");
    end

    initial begin
        reset        = 1'b1;
        step_enable  = 1'b0;
        hidden_spikes = 16'd0;
        checks       = 0;
        failures     = 0;
        step         = 0;
        address      = 0;
        held_membranes = 320'd0;
        held_counts   = 40'd0;

        for (index = 0; index < 160; index = index + 1)
            hidden_reference[index] = 8'hxx;

        for (index = 0; index < 100; index = index + 1) begin
            current_reference[index]  = 32'hxxxxxxxx;
            membrane_reference[index] = 32'hxxxxxxxx;
            spike_reference[index]    = 8'hxx;
        end

        for (index = 0; index < 10; index = index + 1) begin
            count_reference[index] = 8'hxx;
            running_counts[index] = 0;
        end

        prediction_reference[0] = 8'hxx;

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_lif_spikes.mem",
            hidden_reference
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/output_test_currents.mem",
            current_reference
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/output_lif_membranes.mem",
            membrane_reference
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/output_lif_spikes.mem",
            spike_reference
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/output_test_counts.mem",
            count_reference
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/output_test_prediction.mem",
            prediction_reference
        );

        for (index = 0; index < 160; index = index + 1)
            if (
                hidden_reference[index] !== 8'd0
                && hidden_reference[index] !== 8'd1
            )
                $fatal(1, "Invalid/missing hidden spike %0d", index);

        for (index = 0; index < 100; index = index + 1) begin
            if (
                (^current_reference[index] === 1'bx)
                || (^membrane_reference[index] === 1'bx)
                || (
                    spike_reference[index] !== 8'd0
                    && spike_reference[index] !== 8'd1
                )
            )
                $fatal(1, "Invalid/missing output reference %0d", index);
        end

        for (index = 0; index < 10; index = index + 1)
            if (
                (^count_reference[index] === 1'bx)
                || count_reference[index] > 8'd10
            )
                $fatal(1, "Invalid/missing count reference %0d", index);

        if (
            (^prediction_reference[0] === 1'bx)
            || prediction_reference[0] > 8'd9
        )
            $fatal(1, "Invalid/missing prediction reference.");

        // Check that parameter ROMs loaded completely.
        for (index = 0; index < 160; index = index + 1)
            if (^dut.weights[index] === 1'bx)
                $fatal(1, "Missing output weight %0d", index);

        for (index = 0; index < 10; index = index + 1)
            if (^dut.biases[index] === 1'bx)
                $fatal(1, "Missing output bias %0d", index);

        repeat (2) @(posedge clk);
        #1;

        if (
            membranes_out !== 320'd0
            || counts_out !== 40'd0
            || done !== 1'b0
        )
            $fatal(1, "Initial reset failed.");

        @(negedge clk);
        reset = 1'b0;

        $display("==============================================");
        $display("OUTPUT LAYER: VERILOG VS PYTHON");
        $display("==============================================");

        for (step = 0; step < 10; step = step + 1) begin

            @(negedge clk);

            if (ready !== 1'b1)
                $fatal(1, "Not ready for step %0d", step + 1);

            for (neuron = 0; neuron < 16; neuron = neuron + 1)
                hidden_spikes[neuron] =
                    hidden_reference[step*16 + neuron][0];

            step_enable = 1'b1;

            @(posedge clk);
            #1;

            if (step_valid !== 1'b1 || ready !== 1'b0)
                $fatal(1, "LIF handshake failed.");

            for (neuron = 0; neuron < 10; neuron = neuron + 1) begin
                address = step*10 + neuron;
                checks = checks + 1;

                if (
                    currents_out[neuron*32 +: 32] !== current_reference[address]
                    || membranes_out[neuron*32 +: 32] !== membrane_reference[address]
                    || spikes_out[neuron] !== spike_reference[address][0]
                ) begin
                    failures = failures + 1;
                    $display(
                        "FAIL step=%0d digit=%0d current=%0d expected_current=%0d membrane=%0d expected_membrane=%0d spike=%b expected_spike=%b",
                        step + 1,
                        neuron,
                        $signed(currents_out[neuron*32 +: 32]),
                        $signed(current_reference[address]),
                        $signed(membranes_out[neuron*32 +: 32]),
                        $signed(membrane_reference[address]),
                        spikes_out[neuron],
                        spike_reference[address][0]
                    );
                end

                running_counts[neuron] =
                    running_counts[neuron] + spike_reference[address];
            end

            held_membranes = membranes_out;

            // Next clock commits the spike counts.
            @(negedge clk);
            step_enable = 1'b0;

            @(posedge clk);
            #1;

            for (neuron = 0; neuron < 10; neuron = neuron + 1)
                if (
                    counts_out[neuron*4 +: 4]
                    !== running_counts[neuron][3:0]
                )
                    $fatal(
                        1, "Count mismatch: step=%0d digit=%0d",
                        step + 1, neuron
                    );

            if (
                membranes_out !== held_membranes
                || step_valid !== 1'b0
                || spikes_out !== 10'd0
            )
                $fatal(1, "Membrane hold/valid check failed.");

            if (step < 9 && done !== 1'b0)
                $fatal(1, "Premature completion.");

            $display(
                "CHECKED step=%0d prediction_so_far=%0d",
                step + 1, prediction
            );
        end

        if (done !== 1'b1 || ready !== 1'b0)
            $fatal(1, "Final completion handshake failed.");

        for (neuron = 0; neuron < 10; neuron = neuron + 1) begin
            if (
                counts_out[neuron*4 +: 4]
                !== count_reference[neuron][3:0]
            )
                $fatal(1, "Final count mismatch for digit %0d", neuron);

            $display(
                "Digit %0d: %0d spikes",
                neuron, counts_out[neuron*4 +: 4]
            );
        end

        if (prediction !== prediction_reference[0][3:0])
            $fatal(1, "Final prediction mismatch.");

        $display("Predicted digit: %0d", prediction);

        // Further step requests must be ignored after 10 steps.
        held_membranes = membranes_out;
        held_counts = counts_out;

        @(negedge clk);
        step_enable = 1'b1;
        hidden_spikes = 16'hffff;

        repeat (3) @(posedge clk);
        #1;

        if (
            membranes_out !== held_membranes
            || counts_out !== held_counts
            || step_valid !== 1'b0
            || done !== 1'b1
        )
            $fatal(1, "State changed after completion.");

        @(negedge clk);
        step_enable = 1'b0;
        reset = 1'b1;

        @(posedge clk);
        #1;

        if (
            membranes_out !== 320'd0
            || counts_out !== 40'd0
            || done !== 1'b0
            || prediction !== 4'd0
        )
            $fatal(1, "Final reset failed.");

        $display("==============================================");
        $display("Current/membrane/spike triples checked: %0d", checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL OUTPUT LAYER TESTS PASSED");
        else
            $fatal(1, "Output-layer verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule