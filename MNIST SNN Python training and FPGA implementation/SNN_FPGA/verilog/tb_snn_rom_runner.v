`timescale 1ns / 1ps

module tb_snn_rom_runner;

    localparam BASE =
        "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/";

    reg clk;
    reg reset;
    reg start;
    reg [4:0] image_index;

    wire busy;
    wire done;
    wire [3:0] prediction;
    wire [39:0] counts_out;

    reg [7:0] labels [0:19];
    reg [7:0] prediction_ref [0:19];
    reg [7:0] count_ref [0:199];

    reg [3:0] held_prediction;
    reg [39:0] held_counts;

    integer image;
    integer digit;
    integer index;
    integer failures;
    integer matches;
    integer correct;
    integer cycles;
    integer total_count_checks;

    snn_rom_runner dut (
        .clk         (clk),
        .reset       (reset),
        .start       (start),
        .image_index (image_index),
        .busy        (busy),
        .done        (done),
        .prediction  (prediction),
        .counts_out  (counts_out)
    );

    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    initial begin
        #40000000;
        $fatal(1, "ROM-runner test timed out.");
    end

    initial begin
        reset      = 1'b1;
        start      = 1'b0;
        image_index = 5'd0;

        failures = 0;
        matches = 0;
        correct = 0;
        cycles = 0;
        total_count_checks = 0;

        $readmemh({BASE, "multi_labels.mem"}, labels);
        $readmemh({BASE, "multi_predictions.mem"}, prediction_ref);
        $readmemh({BASE, "multi_counts.mem"}, count_ref);

        // Check references before starting.
        for (index = 0; index < 20; index = index + 1)
            if (
                (^labels[index] === 1'bx)
                || (^prediction_ref[index] === 1'bx)
                || labels[index] > 8'd9
                || prediction_ref[index] > 8'd9
            )
                $fatal(1, "Missing/invalid image reference %0d", index);

        for (index = 0; index < 200; index = index + 1)
            if (
                (^count_ref[index] === 1'bx)
                || count_ref[index] > 8'd10
            )
                $fatal(1, "Missing/invalid count reference %0d", index);

        // Check ROMs loaded, including the output-layer ROMs.
        for (index = 0; index < 15680; index = index + 1)
            if (
                (^dut.image_rom[index] === 1'bx)
                || dut.image_rom[index] > 16'd32768
            )
                $fatal(1, "Invalid image ROM entry %0d", index);

        for (index = 0; index < 12544; index = index + 1)
            if (^dut.weight_rom[index] === 1'bx)
                $fatal(1, "Missing hidden weight %0d", index);

        for (index = 0; index < 16; index = index + 1)
            if (^dut.bias_rom[index] === 1'bx)
                $fatal(1, "Missing hidden bias %0d", index);

        for (index = 0; index < 160; index = index + 1)
            if (^dut.network.output_layer.weights[index] === 1'bx)
                $fatal(1, "Missing output weight %0d", index);

        for (index = 0; index < 10; index = index + 1)
            if (^dut.network.output_layer.biases[index] === 1'bx)
                $fatal(1, "Missing output bias %0d", index);

        repeat (2) @(posedge clk);
        #2;

        if (
            busy !== 1'b0
            || done !== 1'b0
            || prediction !== 4'd0
            || counts_out !== 40'd0
        )
            $fatal(1, "Initial reset failed.");

        @(negedge clk);
        reset = 1'b0;

        $display("==============================================");
        $display("AUTOMATIC ROM-LOADED SNN: 20 IMAGES");
        $display("==============================================");

        // No external reset between images:
        // the runner must reset its internal network itself.
        for (image = 0; image < 20; image = image + 1) begin
            @(negedge clk);
            image_index = image;
            start = 1'b1;

            @(posedge clk);
            #2;

            if (busy !== 1'b1 || done !== 1'b0)
                $fatal(1, "Start failed for image %0d", image);

            @(negedge clk);
            start = 1'b0;

            // Change the external selector while busy.
            // The selected image must have been latched.
            image_index = 5'd31;

            cycles = 0;

            while (done !== 1'b1) begin
                @(posedge clk);
                #2;
                cycles = cycles + 1;

                if (cycles > 30000)
                    $fatal(1, "Per-image timeout: image %0d", image);
            end

            if (busy !== 1'b0)
                $fatal(1, "Busy did not clear.");

            for (digit = 0; digit < 10; digit = digit + 1) begin
                total_count_checks = total_count_checks + 1;

                if (
                    counts_out[digit*4 +: 4]
                    !== count_ref[image*10 + digit][3:0]
                ) begin
                    failures = failures + 1;
                    $display(
                        "FAIL count image=%0d digit=%0d got=%0d expected=%0d",
                        image,
                        digit,
                        counts_out[digit*4 +: 4],
                        count_ref[image*10 + digit]
                    );
                end
            end

            if (prediction !== prediction_ref[image][3:0]) begin
                failures = failures + 1;
                $display("FAIL prediction image=%0d", image);
            end
            else
                matches = matches + 1;

            if (prediction === labels[image][3:0])
                correct = correct + 1;

            $display(
                "IMAGE %0d label=%0d prediction=%0d Python=%0d cycles=%0d",
                image,
                labels[image],
                prediction,
                prediction_ref[image],
                cycles
            );

            held_prediction = prediction;
            held_counts = counts_out;

            repeat (3) @(posedge clk);
            #2;

            if (
                prediction !== held_prediction
                || counts_out !== held_counts
                || done !== 1'b1
                || busy !== 1'b0
            )
                $fatal(1, "Completed result did not hold.");
        end

        // Invalid image selection must not start a run.
        @(negedge clk);
        image_index = 5'd31;
        start = 1'b1;

        @(posedge clk);
        #2;

        if (
            busy !== 1'b0
            || done !== 1'b1
            || counts_out !== held_counts
        )
            $fatal(1, "Invalid image index was accepted.");

        @(negedge clk);
        start = 1'b0;

        $display("==============================================");
        $display("Images tested: 20");
        $display("Predictions matching Python: %0d", matches);
        $display("Correct classifications: %0d", correct);
        $display("Final spike counts checked: %0d", total_count_checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("ALL ROM RUNNER TESTS PASSED");
        else
            $fatal(1, "ROM-runner verification failed.");

        $display("==============================================");
        $finish;
    end

endmodule