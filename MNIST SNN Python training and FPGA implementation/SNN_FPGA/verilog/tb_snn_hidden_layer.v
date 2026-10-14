`timescale 1ns / 1ps

module tb_snn_hidden_layer;

    reg clk;
    reg reset;
    reg current_start;
    reg [3:0] neuron_index;
    reg signed [15:0] bias_in;
    reg input_valid;
    reg [15:0] pixel_in;
    reg signed [15:0] weight_in;
    reg step_enable;

    wire current_busy;
    wire current_loaded;
    wire all_currents_ready;
    wire [15:0] spikes_out;
    wire [511:0] membranes_out;
    wire step_valid;

    reg [15:0] pixels [0:783];
    reg signed [15:0] weights [0:12543];
    reg signed [15:0] biases [0:15];
    reg signed [31:0] expected_currents [0:15];
    reg signed [31:0] expected_membranes [0:159];
    reg [7:0] expected_spikes [0:159];

    reg [511:0] held_membranes;

    integer index;
    integer neuron;
    integer pixel;
    integer step;
    integer address;
    integer checks;
    integer failures;

    snn_hidden_layer dut (
        .clk                (clk),
        .reset              (reset),
        .current_start      (current_start),
        .neuron_index       (neuron_index),
        .bias_in            (bias_in),
        .input_valid        (input_valid),
        .pixel_in           (pixel_in),
        .weight_in          (weight_in),
        .current_busy       (current_busy),
        .current_loaded     (current_loaded),
        .all_currents_ready (all_currents_ready),
        .step_enable        (step_enable),
        .spikes_out         (spikes_out),
        .membranes_out      (membranes_out),
        .step_valid         (step_valid)
    );

    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    initial begin
        #2000000;
        $fatal(1, "Timeout: hidden-layer test did not finish.");
    end

    initial begin
        reset        = 1'b1;
        current_start = 1'b0;
        neuron_index = 4'd0;
        bias_in      = 16'sd0;
        input_valid  = 1'b0;
        pixel_in     = 16'd0;
        weight_in    = 16'sd0;
        step_enable  = 1'b0;

        checks   = 0;
        failures = 0;

        for (index = 0; index < 784; index = index + 1)
            pixels[index] = 16'hxxxx;

        for (index = 0; index < 12544; index = index + 1)
            weights[index] = 16'hxxxx;

        for (index = 0; index < 16; index = index + 1) begin
            biases[index] = 16'hxxxx;
            expected_currents[index] = 32'hxxxxxxxx;
        end

        for (index = 0; index < 160; index = index + 1) begin
            expected_membranes[index] = 32'hxxxxxxxx;
            expected_spikes[index] = 8'hxx;
        end

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/mnist_test_image.mem",
            pixels
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc1_weights.mem",
            weights
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/fc1_biases.mem",
            biases
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_test_currents.mem",
            expected_currents
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_lif_membranes.mem",
            expected_membranes
        );
        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_lif_spikes.mem",
            expected_spikes
        );

        for (index = 0; index < 784; index = index + 1) begin
            if (^pixels[index] === 1'bx)
                $fatal(1, "Missing pixel %0d", index);
            if (pixels[index] > 16'd32768)
                $fatal(1, "Invalid pixel %0d", index);
        end

        for (index = 0; index < 12544; index = index + 1)
            if (^weights[index] === 1'bx)
                $fatal(1, "Missing weight %0d", index);

        for (index = 0; index < 16; index = index + 1)
            if (
                (^biases[index] === 1'bx)
                || (^expected_currents[index] === 1'bx)
            )
                $fatal(1, "Missing parameter/reference %0d", index);

        for (index = 0; index < 160; index = index + 1) begin
            if (
                (^expected_membranes[index] === 1'bx)
                || (^expected_spikes[index] === 1'bx)
            )
                $fatal(1, "Missing LIF reference %0d", index);

            if (
                expected_spikes[index] !== 8'd0
                && expected_spikes[index] !== 8'd1
            )
                $fatal(1, "Invalid spike reference %0d", index);
        end

        repeat (2) @(posedge clk);
        #1;

        if (
            membranes_out !== 512'd0
            || spikes_out !== 16'd0
            || step_valid !== 1'b0
            || all_currents_ready !== 1'b0
        )
            $fatal(1, "Initial reset failed.");

        @(negedge clk);
        reset = 1'b0;

        // A premature step request must be ignored.
        step_enable = 1'b1;

        @(posedge clk);
        #1;

        if (
            membranes_out !== 512'd0
            || step_valid !== 1'b0
        )
            $fatal(1, "Advanced before currents were ready.");

        @(negedge clk);
        step_enable = 1'b0;

        $display("==============================================");
        $display("COMPLETE HIDDEN-LAYER VERIFICATION");
        $display("==============================================");

        // Calculate and store each hidden current.
        for (neuron = 0; neuron < 16; neuron = neuron + 1) begin
            @(negedge clk);
            neuron_index  = neuron;
            bias_in       = biases[neuron];
            current_start = 1'b1;
            input_valid   = 1'b0;

            @(posedge clk);
            #1;

            if (current_busy !== 1'b1)
                $fatal(1, "MAC did not start.");

            @(negedge clk);
            current_start = 1'b0;

            for (pixel = 0; pixel < 784; pixel = pixel + 1) begin
                @(negedge clk);
                input_valid = 1'b1;
                pixel_in    = pixels[pixel];
                weight_in   = weights[neuron*784 + pixel];

                @(posedge clk);
                #1;
            end

            // Stop feeding pairs. On the next rising edge,
            // the layer captures the completed MAC current.
            @(negedge clk);
            input_valid = 1'b0;

            @(posedge clk);
            #1;

            if (
                current_loaded !== 1'b1
                || dut.stored_currents[neuron] !== expected_currents[neuron]
            )
                $fatal(1, "Current storage failed for neuron %0d", neuron);

            if (membranes_out !== 512'd0)
                $fatal(1, "Membranes changed during current loading.");

            $display(
                "LOADED neuron=%0d current=%0d",
                neuron, $signed(dut.stored_currents[neuron])
            );
        end

        if (all_currents_ready !== 1'b1)
            $fatal(1, "Not all currents are ready.");

        // Advance all 16 neurons together for 10 SNN steps.
        for (step = 0; step < 10; step = step + 1) begin
            @(negedge clk);
            step_enable = 1'b1;

            @(posedge clk);
            #1;

            if (step_valid !== 1'b1)
                $fatal(1, "Missing valid signal at step %0d", step + 1);

            for (neuron = 0; neuron < 16; neuron = neuron + 1) begin
                address = step*16 + neuron;
                checks = checks + 1;

                if (
                    membranes_out[neuron*32 +: 32] !== expected_membranes[address]
                    || spikes_out[neuron] !== expected_spikes[address][0]
                ) begin
                    failures = failures + 1;

                    $display(
                        "FAIL step=%0d neuron=%0d membrane=%0d expected=%0d spike=%b expected=%b",
                        step + 1,
                        neuron,
                        $signed(membranes_out[neuron*32 +: 32]),
                        $signed(expected_membranes[address]),
                        spikes_out[neuron],
                        expected_spikes[address][0]
                    );
                end
            end

            $display(
                "CHECKED step=%0d spikes[15:0]=%b",
                step + 1, spikes_out
            );

            held_membranes = membranes_out;

            @(negedge clk);
            step_enable = 1'b0;

            @(posedge clk);
            #1;

            if (
                membranes_out !== held_membranes
                || step_valid !== 1'b0
                || spikes_out !== 16'd0
            )
                $fatal(1, "Disabled-state check failed.");
        end

        // Verify that a new image reset clears every membrane
        // and invalidates the stored-current readiness flags.
        @(negedge clk);
        reset = 1'b1;

        @(posedge clk);
        #1;

        if (
            membranes_out !== 512'd0
            || spikes_out !== 16'd0
            || all_currents_ready !== 1'b0
            || step_valid !== 1'b0
        )
            $fatal(1, "Final reset failed.");

        $display("==============================================");
        $display("Membrane/spike pairs checked: %0d", checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL HIDDEN LAYER TESTS PASSED");
        else
            $fatal(1, "Hidden-layer verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule