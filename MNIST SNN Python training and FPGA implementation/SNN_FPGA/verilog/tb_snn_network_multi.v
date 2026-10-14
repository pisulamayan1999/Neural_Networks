`timescale 1ns / 1ps

module tb_snn_network_multi;

    localparam integer NUM_IMAGES = 20;

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

    reg [15:0] pixels [0:NUM_IMAGES*784-1];
    reg signed [15:0] weights [0:12543];
    reg signed [15:0] biases [0:15];

    reg [7:0] labels [0:NUM_IMAGES-1];
    reg [7:0] prediction_ref [0:NUM_IMAGES-1];
    reg [7:0] count_ref [0:NUM_IMAGES*10-1];

    reg signed [31:0] hidden_mem_ref [0:NUM_IMAGES*160-1];
    reg [7:0] hidden_spike_ref [0:NUM_IMAGES*160-1];

    reg signed [31:0] output_current_ref [0:NUM_IMAGES*100-1];
    reg signed [31:0] output_mem_ref [0:NUM_IMAGES*100-1];
    reg [7:0] output_spike_ref [0:NUM_IMAGES*100-1];

    integer image_index;
    integer index;
    integer neuron;
    integer pixel;

    integer h;
    integer o;
    integer hidden_address;
    integer output_address;

    integer hidden_steps;
    integer output_steps;
    integer hidden_checks;
    integer output_checks;

    integer failures;
    integer matched_predictions;
    integer correct_predictions;

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

    // Expected simulated runtime is about 10 milliseconds.
    // This is simulation time, not a wall-clock deadline.
    initial begin
        #20000000;
        $fatal(1, "Multi-image test timed out.");
    end

    // Check every valid intermediate result.
    always @(posedge clk) begin
        #1;

        if (!reset) begin
            if (hidden_valid === 1'b1) begin
                if (hidden_steps >= 10)
                    $fatal(1, "Too many hidden steps.");

                for (h = 0; h < 16; h = h + 1) begin
                    hidden_address =
                        image_index*160 + hidden_steps*16 + h;

                    hidden_checks = hidden_checks + 1;

                    if (
                        hidden_membranes[h*32 +: 32]
                            !== hidden_mem_ref[hidden_address]
                        || hidden_spikes[h]
                            !== hidden_spike_ref[hidden_address][0]
                    ) begin
                        failures = failures + 1;

                        $display(
                            "FAIL hidden image=%0d step=%0d neuron=%0d membrane=%0d expected=%0d spike=%b expected=%b",
                            image_index,
                            hidden_steps + 1,
                            h,
                            $signed(hidden_membranes[h*32 +: 32]),
                            $signed(hidden_mem_ref[hidden_address]),
                            hidden_spikes[h],
                            hidden_spike_ref[hidden_address][0]
                        );
                    end
                end

                hidden_steps = hidden_steps + 1;
            end

            if (output_valid === 1'b1) begin
                if (output_steps >= 10)
                    $fatal(1, "Too many output steps.");

                if (hidden_steps != output_steps + 1)
                    $fatal(1, "Hidden/output step ordering failed.");

                for (o = 0; o < 10; o = o + 1) begin
                    output_address =
                        image_index*100 + output_steps*10 + o;

                    output_checks = output_checks + 1;

                    if (
                        output_currents[o*32 +: 32]
                            !== output_current_ref[output_address]
                        || output_membranes[o*32 +: 32]
                            !== output_mem_ref[output_address]
                        || output_spikes[o]
                            !== output_spike_ref[output_address][0]
                    ) begin
                        failures = failures + 1;

                        $display(
                            "FAIL output image=%0d step=%0d digit=%0d current=%0d expected_current=%0d membrane=%0d expected_membrane=%0d spike=%b expected=%b",
                            image_index,
                            output_steps + 1,
                            o,
                            $signed(output_currents[o*32 +: 32]),
                            $signed(output_current_ref[output_address]),
                            $signed(output_membranes[o*32 +: 32]),
                            $signed(output_mem_ref[output_address]),
                            output_spikes[o],
                            output_spike_ref[output_address][0]
                        );
                    end
                end

                output_steps = output_steps + 1;
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

        image_index   = 0;
        hidden_steps  = 0;
        output_steps  = 0;
        hidden_checks = 0;
        output_checks = 0;
        hidden_address = 0;
        output_address = 0;

        failures           = 0;
        matched_predictions = 0;
        correct_predictions = 0;

        $readmemh({BASE, "multi_images.mem"}, pixels);
        $readmemh({BASE, "fc1_weights.mem"}, weights);
        $readmemh({BASE, "fc1_biases.mem"}, biases);

        $readmemh({BASE, "multi_labels.mem"}, labels);
        $readmemh({BASE, "multi_predictions.mem"}, prediction_ref);
        $readmemh({BASE, "multi_counts.mem"}, count_ref);

        $readmemh(
            {BASE, "multi_hidden_membranes.mem"}, hidden_mem_ref
        );
        $readmemh(
            {BASE, "multi_hidden_spikes.mem"}, hidden_spike_ref
        );
        $readmemh(
            {BASE, "multi_output_currents.mem"}, output_current_ref
        );
        $readmemh(
            {BASE, "multi_output_membranes.mem"}, output_mem_ref
        );
        $readmemh(
            {BASE, "multi_output_spikes.mem"}, output_spike_ref
        );

        // Validate all loaded test data before using it.
        for (index = 0; index < NUM_IMAGES*784; index = index + 1)
            if (
                (^pixels[index] === 1'bx)
                || pixels[index] > 16'd32768
            )
                $fatal(1, "Invalid/missing pixel %0d", index);

        for (index = 0; index < 12544; index = index + 1)
            if (^weights[index] === 1'bx)
                $fatal(1, "Missing hidden weight %0d", index);

        for (index = 0; index < 16; index = index + 1)
            if (^biases[index] === 1'bx)
                $fatal(1, "Missing hidden bias %0d", index);

        for (index = 0; index < NUM_IMAGES; index = index + 1)
            if (
                (^labels[index] === 1'bx)
                || (^prediction_ref[index] === 1'bx)
                || labels[index] > 8'd9
                || prediction_ref[index] > 8'd9
            )
                $fatal(1, "Invalid label/prediction %0d", index);

        for (index = 0; index < NUM_IMAGES*10; index = index + 1)
            if (
                (^count_ref[index] === 1'bx)
                || count_ref[index] > 8'd10
            )
                $fatal(1, "Invalid count reference %0d", index);

        for (index = 0; index < NUM_IMAGES*160; index = index + 1)
            if (
                (^hidden_mem_ref[index] === 1'bx)
                || (
                    hidden_spike_ref[index] !== 8'd0
                    && hidden_spike_ref[index] !== 8'd1
                )
            )
                $fatal(1, "Invalid hidden reference %0d", index);

        for (index = 0; index < NUM_IMAGES*100; index = index + 1)
            if (
                (^output_current_ref[index] === 1'bx)
                || (^output_mem_ref[index] === 1'bx)
                || (
                    output_spike_ref[index] !== 8'd0
                    && output_spike_ref[index] !== 8'd1
                )
            )
                $fatal(1, "Invalid output reference %0d", index);

        $display("==============================================");
        $display("20-IMAGE FULL SNN VERIFICATION");
        $display("==============================================");

        for (
            image_index = 0;
            image_index < NUM_IMAGES;
            image_index = image_index + 1
        ) begin

            // Reset the complete network for every image.
            @(negedge clk);
            reset         = 1'b1;
            current_start = 1'b0;
            input_valid   = 1'b0;
            run_start     = 1'b0;

            hidden_steps = 0;
            output_steps = 0;

            repeat (2) @(posedge clk);
            #2;

            if (
                hidden_membranes !== 512'd0
                || output_membranes !== 320'd0
                || counts_out !== 40'd0
                || all_currents_ready !== 1'b0
                || done !== 1'b0
                || run_busy !== 1'b0
            )
                $fatal(1, "Reset failed for image %0d", image_index);

            @(negedge clk);
            reset = 1'b0;

            // Load all hidden currents for this image.
            for (neuron = 0; neuron < 16; neuron = neuron + 1) begin
                @(negedge clk);
                current_start = 1'b1;
                neuron_index = neuron;
                bias_in = biases[neuron];

                @(posedge clk);
                #2;

                if (current_busy !== 1'b1)
                    $fatal(1, "MAC failed to start.");

                @(negedge clk);
                current_start = 1'b0;

                for (pixel = 0; pixel < 784; pixel = pixel + 1) begin
                    @(negedge clk);
                    input_valid = 1'b1;
                    pixel_in = pixels[image_index*784 + pixel];
                    weight_in = weights[neuron*784 + pixel];

                    @(posedge clk);
                    #2;
                end

                @(negedge clk);
                input_valid = 1'b0;

                @(posedge clk);
                #2;

                if (current_loaded !== 1'b1)
                    $fatal(1, "Hidden current was not stored.");
            end

            @(posedge clk);
            #2;

            if (all_currents_ready !== 1'b1)
                $fatal(1, "Currents not ready.");

            @(negedge clk);
            run_start = 1'b1;

            @(posedge clk);
            #2;

            if (run_busy !== 1'b1)
                $fatal(1, "Network failed to start.");

            @(negedge clk);
            run_start = 1'b0;

            wait (done === 1'b1);
            @(negedge clk);

            if (hidden_steps != 10 || output_steps != 10)
                $fatal(1, "Wrong step count for image %0d", image_index);

            for (neuron = 0; neuron < 10; neuron = neuron + 1)
                if (
                    counts_out[neuron*4 +: 4]
                    !== count_ref[image_index*10 + neuron][3:0]
                ) begin
                    failures = failures + 1;
                    $display(
                        "FAIL count image=%0d digit=%0d",
                        image_index, neuron
                    );
                end

            if (prediction !== prediction_ref[image_index][3:0]) begin
                failures = failures + 1;
                $display(
                    "FAIL prediction image=%0d got=%0d expected=%0d",
                    image_index, prediction, prediction_ref[image_index]
                );
            end
            else
                matched_predictions = matched_predictions + 1;

            if (prediction === labels[image_index][3:0])
                correct_predictions = correct_predictions + 1;

            $display(
                "IMAGE %0d label=%0d prediction=%0d Python=%0d",
                image_index,
                labels[image_index],
                prediction,
                prediction_ref[image_index]
            );

            repeat (4) @(posedge clk);
            #2;

            if (
                run_busy !== 1'b0
                || hidden_steps != 10
                || output_steps != 10
            )
                $fatal(1, "Controller did not stop correctly.");
        end

        if (
            hidden_checks != NUM_IMAGES*160
            || output_checks != NUM_IMAGES*100
        )
            $fatal(1, "Incorrect total number of checks.");

        $display("==============================================");
        $display("Images tested: %0d", NUM_IMAGES);
        $display("Predictions matching Python: %0d", matched_predictions);
        $display("Correct classifications: %0d", correct_predictions);
        $display("Hidden pairs checked: %0d", hidden_checks);
        $display("Output triples checked: %0d", output_checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL MULTI-IMAGE NETWORK TESTS PASSED");
        else
            $fatal(1, "Multi-image verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule