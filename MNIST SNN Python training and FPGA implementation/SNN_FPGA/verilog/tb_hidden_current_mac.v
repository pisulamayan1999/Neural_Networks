`timescale 1ns / 1ps

module tb_hidden_current_mac;

    reg clk;
    reg reset;
    reg start;
    reg input_valid;

    reg signed [15:0] bias_in;
    reg        [15:0] pixel_in;
    reg signed [15:0] weight_in;

    wire busy;
    wire done;
    wire signed [47:0] accumulator_out;
    wire signed [31:0] current_out;

    reg        [15:0] pixels [0:783];
    reg signed [15:0] weights [0:12543];
    reg signed [15:0] biases [0:15];

    reg signed [47:0] expected_accumulators [0:15];
    reg signed [31:0] expected_currents [0:15];

    reg signed [47:0] held_accumulator;

    integer index;
    integer neuron;
    integer pixel;
    integer failures;
    integer checks;

    hidden_current_mac dut (
        .clk             (clk),
        .reset           (reset),
        .start           (start),
        .bias_in         (bias_in),
        .input_valid     (input_valid),
        .pixel_in        (pixel_in),
        .weight_in       (weight_in),
        .busy            (busy),
        .done            (done),
        .accumulator_out (accumulator_out),
        .current_out     (current_out)
    );

    // 25 MHz.
    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    // Prevent an unexpected stalled test from running forever.
    initial begin
        #2000000;
        $fatal(1, "Timeout: hidden-current test did not finish.");
    end

    initial begin
        reset      = 1'b1;
        start      = 1'b0;
        input_valid = 1'b0;
        bias_in    = 16'sd0;
        pixel_in   = 16'd0;
        weight_in  = 16'sd0;

        failures = 0;
        checks   = 0;

        for (index = 0; index < 784; index = index + 1)
            pixels[index] = 16'hxxxx;

        for (index = 0; index < 12544; index = index + 1)
            weights[index] = 16'hxxxx;

        for (index = 0; index < 16; index = index + 1) begin
            biases[index] = 16'hxxxx;
            expected_accumulators[index] = 48'hxxxxxxxxxxxx;
            expected_currents[index] = 32'hxxxxxxxx;
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
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_test_accumulators.mem",
            expected_accumulators
        );

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/hidden_test_currents.mem",
            expected_currents
        );

        for (index = 0; index < 784; index = index + 1) begin
            if (^pixels[index] === 1'bx)
                $fatal(1, "Missing pixel at address %0d", index);

            if (pixels[index] > 16'd32768)
                $fatal(1, "Pixel outside expected range at address %0d", index);
        end

        for (index = 0; index < 12544; index = index + 1) begin
            if (^weights[index] === 1'bx)
                $fatal(1, "Missing weight at address %0d", index);
        end

        for (index = 0; index < 16; index = index + 1) begin
            if (
                (^biases[index] === 1'bx)
                || (^expected_accumulators[index] === 1'bx)
                || (^expected_currents[index] === 1'bx)
            )
                $fatal(1, "Missing reference for neuron %0d", index);
        end

        repeat (2) @(posedge clk);
        #1;

        if (
            busy !== 1'b0
            || done !== 1'b0
            || current_out !== 32'sd0
        )
            $fatal(1, "Reset failed.");

        @(negedge clk);
        reset = 1'b0;

        $display("==============================================");
        $display("HIDDEN CURRENT: VERILOG VS PYTHON");
        $display("MNIST TEST IMAGE 0: DIGIT 7");
        $display("==============================================");

        // Reuse the same calculator for all 16 neurons.
        // No reset between neurons: start must reload the bias
        // and clear the previous accumulation.
        for (neuron = 0; neuron < 16; neuron = neuron + 1) begin

            @(negedge clk);
            start       = 1'b1;
            input_valid = 1'b0;
            bias_in     = biases[neuron];

            @(posedge clk);
            #1;

            if (busy !== 1'b1 || done !== 1'b0)
                $fatal(1, "Start handshake failed for neuron %0d", neuron);

            @(negedge clk);
            start = 1'b0;

            for (pixel = 0; pixel < 784; pixel = pixel + 1) begin

                // Occasionally insert an invalid input cycle.
                // The accumulator must hold its previous value.
                if ((pixel % 100) == 0) begin
                    held_accumulator = dut.accumulator;

                    @(negedge clk);
                    input_valid = 1'b0;
                    pixel_in    = 16'd32768;
                    weight_in   = -16'sd1234;

                    @(posedge clk);
                    #1;

                    if (
                        dut.accumulator !== held_accumulator
                        || busy !== 1'b1
                        || done !== 1'b0
                    )
                        $fatal(
                            1,
                            "Invalid input cycle changed state: neuron=%0d pixel=%0d",
                            neuron, pixel
                        );
                end

                @(negedge clk);
                input_valid = 1'b1;
                pixel_in    = pixels[pixel];

                // Export ordering:
                // all 784 weights of neuron 0, then neuron 1...
                weight_in = weights[neuron * 784 + pixel];

                @(posedge clk);
                #1;

                if (pixel < 783) begin
                    if (busy !== 1'b1 || done !== 1'b0)
                        $fatal(
                            1,
                            "Unexpected early completion: neuron=%0d pixel=%0d",
                            neuron, pixel
                        );
                end
                else begin
                    checks = checks + 1;

                    if (
                        busy !== 1'b0
                        || done !== 1'b1
                        || accumulator_out !== expected_accumulators[neuron]
                        || current_out !== expected_currents[neuron]
                    ) begin
                        failures = failures + 1;

                        $display(
                            "FAIL neuron=%0d current=%0d expected=%0d accumulator=%0d expected_accumulator=%0d",
                            neuron,
                            $signed(current_out),
                            $signed(expected_currents[neuron]),
                            $signed(accumulator_out),
                            $signed(expected_accumulators[neuron])
                        );
                    end
                    else begin
                        $display(
                            "PASS neuron=%0d current=%0d",
                            neuron, $signed(current_out)
                        );
                    end
                end
            end

            @(negedge clk);
            input_valid = 1'b0;

            @(posedge clk);
            #1;

            if (
                done !== 1'b0
                || busy !== 1'b0
                || current_out !== expected_currents[neuron]
            )
                $fatal(1, "Completion pulse/output hold failed.");
        end

        $display("==============================================");
        $display("Hidden currents checked: %0d", checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL HIDDEN CURRENT TESTS PASSED");
        else
            $fatal(1, "Hidden-current verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule