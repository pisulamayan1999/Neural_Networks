`timescale 1ns / 1ps

module tb_snn_network;

    localparam BASE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/";

    reg clk;
    reg reset;
    reg current_start;
    reg [3:0] neuron_index;
    reg signed [15:0] bias_in;
    reg input_valid;
    reg [15:0] pixel_in;
    reg signed [15:0] weight_in;
    reg run_start;

    wire current_busy;
    wire current_loaded;
    wire all_currents_ready;
    wire run_busy;
    wire done;
    wire [3:0] prediction;
    wire [39:0] counts_out;

    wire hidden_valid;
    wire [15:0] hidden_spikes;
    wire [511:0] hidden_membranes;

    wire output_valid;
    wire [319:0] output_currents;
    wire [319:0] output_membranes;
    wire [9:0] output_spikes;

    reg [15:0] pixels [0:783];
    reg signed [15:0] weights [0:12543];
    reg signed [15:0] biases [0:15];

    reg signed [31:0] hidden_mem_ref [0:159];
    reg [7:0] hidden_spike_ref [0:159];

    reg signed [31:0] output_current_ref [0:99];
    reg signed [31:0] output_mem_ref [0:99];
    reg [7:0] output_spike_ref [0:99];

    reg [7:0] count_ref [0:9];
    reg [7:0] prediction_ref [0:0];

    integer neuron;
    integer pixel;
    integer h;
    integer o;
    integer hidden_steps;
    integer output_steps;
    integer hidden_checks;
    integer output_checks;
    integer failures;

    snn_network dut (
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
        .run_start          (run_start),
        .run_busy           (run_busy),
        .done               (done),
        .prediction         (prediction),
        .counts_out         (counts_out),
        .hidden_valid       (hidden_valid),
        .hidden_spikes      (hidden_spikes),
        .hidden_membranes   (hidden_membranes),
        .output_valid       (output_valid),
        .output_currents    (output_currents),
        .output_membranes   (output_membranes),
        .output_spikes      (output_spikes)
    );

    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    initial begin
        #2000000;
        $fatal(1, "Full-network test timed out.");
    end

    // Check each layer exactly when its results become valid.
    always @(posedge clk) begin
        #1;

        if (!reset) begin
            if (hidden_valid === 1'b1) begin
                if (hidden_steps >= 10)
                    $fatal(1, "More than 10 hidden steps.");

                for (h = 0; h < 16; h = h + 1) begin
                    hidden_checks = hidden_checks + 1;

                    if (
                        hidden_membranes[h*32 +: 32]
                            !== hidden_mem_ref[hidden_steps*16 + h]
                        || hidden_spikes[h]
                            !== hidden_spike_ref[hidden_steps*16 + h][0]
                    ) begin
                        failures = failures + 1;
                        $display(
                            "FAIL hidden step=%0d neuron=%0d",
                            hidden_steps + 1, h
                        );
                    end
                end

                hidden_steps = hidden_steps + 1;
            end

            if (output_valid === 1'b1) begin
                if (output_steps >= 10)
                    $fatal(1, "More than 10 output steps.");

                if (hidden_steps != output_steps + 1)
                    $fatal(1, "Layer step ordering failed.");

                for (o = 0; o < 10; o = o + 1) begin
                    output_checks = output_checks + 1;

                    if (
                        output_currents[o*32 +: 32]
                            !== output_current_ref[output_steps*10 + o]
                        || output_membranes[o*32 +: 32]
                            !== output_mem_ref[output_steps*10 + o]
                        || output_spikes[o]
                            !== output_spike_ref[output_steps*10 + o][0]
                    ) begin
                        failures = failures + 1;
                        $display(
                            "FAIL output step=%0d digit=%0d",
                            output_steps + 1, o
                        );
                    end
                end

                output_steps = output_steps + 1;
                $display("CHECKED full SNN step=%0d", output_steps);
            end
        end
    end

    initial begin
        reset         = 1'b1;
        current_start = 1'b0;
        neuron_index  = 4'd0;
        bias_in       = 16'sd0;
        input_valid   = 1'b0;
        pixel_in      = 16'd0;
        weight_in     = 16'sd0;
        run_start     = 1'b0;

        hidden_steps  = 0;
        output_steps  = 0;
        hidden_checks = 0;
        output_checks = 0;
        failures      = 0;

        $readmemh({BASE, "mnist_test_image.mem"}, pixels);
        $readmemh({BASE, "fc1_weights.mem"}, weights);
        $readmemh({BASE, "fc1_biases.mem"}, biases);

        $readmemh({BASE, "hidden_lif_membranes.mem"}, hidden_mem_ref);
        $readmemh({BASE, "hidden_lif_spikes.mem"}, hidden_spike_ref);

        $readmemh({BASE, "output_test_currents.mem"}, output_current_ref);
        $readmemh({BASE, "output_lif_membranes.mem"}, output_mem_ref);
        $readmemh({BASE, "output_lif_spikes.mem"}, output_spike_ref);

        $readmemh({BASE, "output_test_counts.mem"}, count_ref);
        $readmemh({BASE, "output_test_prediction.mem"}, prediction_ref);

        // Missing data retains unknown values. Check it before use.
        for (pixel = 0; pixel < 784; pixel = pixel + 1)
            if (
                (^pixels[pixel] === 1'bx)
                || pixels[pixel] > 16'd32768
            )
                $fatal(1, "Invalid/missing pixel %0d", pixel);

        for (pixel = 0; pixel < 12544; pixel = pixel + 1)
            if (^weights[pixel] === 1'bx)
                $fatal(1, "Missing weight %0d", pixel);

        for (neuron = 0; neuron < 16; neuron = neuron + 1)
            if (^biases[neuron] === 1'bx)
                $fatal(1, "Missing bias %0d", neuron);

        for (pixel = 0; pixel < 160; pixel = pixel + 1)
            if (
                (^hidden_mem_ref[pixel] === 1'bx)
                || (
                    hidden_spike_ref[pixel] !== 8'd0
                    && hidden_spike_ref[pixel] !== 8'd1
                )
            )
                $fatal(1, "Invalid hidden reference %0d", pixel);

        for (pixel = 0; pixel < 100; pixel = pixel + 1)
            if (
                (^output_current_ref[pixel] === 1'bx)
                || (^output_mem_ref[pixel] === 1'bx)
                || (
                    output_spike_ref[pixel] !== 8'd0
                    && output_spike_ref[pixel] !== 8'd1
                )
            )
                $fatal(1, "Invalid output reference %0d", pixel);

        for (neuron = 0; neuron < 10; neuron = neuron + 1)
            if (
                (^count_ref[neuron] === 1'bx)
                || count_ref[neuron] > 8'd10
            )
                $fatal(1, "Invalid count reference %0d", neuron);

        if (
            (^prediction_ref[0] === 1'bx)
            || prediction_ref[0] > 8'd9
        )
            $fatal(1, "Invalid prediction reference.");

        repeat (2) @(posedge clk);

        @(negedge clk);
        reset = 1'b0;

        // Calculate all 16 hidden currents from actual pixels.
        for (neuron = 0; neuron < 16; neuron = neuron + 1) begin
            @(negedge clk);
            current_start = 1'b1;
            neuron_index = neuron;
            bias_in = biases[neuron];

            @(posedge clk);
            #1;

            if (current_busy !== 1'b1)
                $fatal(1, "MAC start failed.");

            @(negedge clk);
            current_start = 1'b0;

            for (pixel = 0; pixel < 784; pixel = pixel + 1) begin
                @(negedge clk);
                input_valid = 1'b1;
                pixel_in = pixels[pixel];
                weight_in = weights[neuron*784 + pixel];

                @(posedge clk);
                #1;
            end

            @(negedge clk);
            input_valid = 1'b0;

            @(posedge clk);
            #1;

            if (current_loaded !== 1'b1)
                $fatal(1, "Current load failed for neuron %0d", neuron);
        end

        // Allow the final load pulse to clear.
        @(posedge clk);
        #1;

        if (all_currents_ready !== 1'b1)
            $fatal(1, "Hidden currents not ready.");

        $display("==============================================");
        $display("FULL 784 -> 16 -> 10 SNN VERIFICATION");
        $display("==============================================");

        @(negedge clk);
        run_start = 1'b1;

        @(posedge clk);
        #1;

        if (run_busy !== 1'b1)
            $fatal(1, "Network did not start.");

        @(negedge clk);
        run_start = 1'b0;

        wait (done === 1'b1);

        // Check on a falling edge, after registered counts settle.
        @(negedge clk);

        if (
            hidden_steps != 10
            || output_steps != 10
            || hidden_checks != 160
            || output_checks != 100
        )
            $fatal(1, "Incorrect number of layer updates.");

        for (neuron = 0; neuron < 10; neuron = neuron + 1) begin
            if (
                counts_out[neuron*4 +: 4]
                !== count_ref[neuron][3:0]
            )
                $fatal(1, "Final count mismatch for digit %0d", neuron);

            $display(
                "Digit %0d: %0d spikes",
                neuron, counts_out[neuron*4 +: 4]
            );
        end

        if (prediction !== prediction_ref[0][3:0])
            $fatal(1, "Final prediction mismatch.");

        // Give the controller time to enter FINISHED.
        repeat (4) @(posedge clk);
        #2;

        if (
            run_busy !== 1'b0
            || hidden_steps != 10
            || output_steps != 10
        )
            $fatal(1, "Controller did not stop correctly.");

        $display("==============================================");
        $display("Hidden pairs checked: %0d", hidden_checks);
        $display("Output triples checked: %0d", output_checks);
        $display("Predicted digit: %0d", prediction);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL FULL NETWORK TESTS PASSED");
        else
            $fatal(1, "Full-network verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule