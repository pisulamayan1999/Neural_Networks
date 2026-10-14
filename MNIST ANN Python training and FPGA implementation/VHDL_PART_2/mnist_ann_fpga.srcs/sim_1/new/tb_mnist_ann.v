`timescale 1ns / 1ps

module tb_mnist_ann;

    // ============================================================
    // Testbench signals
    // ============================================================

    reg clk;
    reg reset;
    reg start;

    wire signed [15:0] pixel_in;

    wire [9:0] pixel_index;

    wire [3:0] predicted_digit;
    wire done;


    // ============================================================
    // Real MNIST image memory
    //
    // 28 x 28 = 784 pixels
    //
    // Each value is stored in Q1.15 format.
    // ============================================================

    reg signed [15:0] image_memory [0:783];


    // ============================================================
    // Load real MNIST image
    // ============================================================

    initial begin

        $readmemh(
            "mnist_image.mem",
            image_memory
        );

    end


    // ============================================================
    // Feed requested pixel into ANN
    //
    // mnist_ann tells us which pixel it currently needs
    // using pixel_index.
    // ============================================================

    assign pixel_in = image_memory[pixel_index];


    // ============================================================
    // Complete ANN
    // ============================================================

    mnist_ann uut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .pixel_in(pixel_in),
        .pixel_index(pixel_index),

        .predicted_digit(predicted_digit),
        .done(done)
    );


    // ============================================================
    // Clock
    //
    // 10 ns period = 100 MHz
    // ============================================================

    initial begin

        clk = 1'b0;

        forever begin
            #5 clk = ~clk;
        end

    end


    // ============================================================
    // Main real-MNIST test
    // ============================================================

    initial begin

        reset = 1'b1;
        start = 1'b0;


        // --------------------------------------------------------
        // Reset network
        // --------------------------------------------------------

        #30;

        reset = 1'b0;

        #20;


        // --------------------------------------------------------
        // Start complete ANN
        // --------------------------------------------------------

        $display("");
        $display("================================================");
        $display("STARTING REAL MNIST IMAGE TEST");
        $display("================================================");
        $display("");

        $display("MNIST test image index = 0");
        $display("Actual digit           = 7");
        $display("Python prediction      = 7");
        $display("Q1.15 reference        = 7");

        $display("");
        $display("Running complete FPGA ANN...");
        $display("");


        start = 1'b1;

        #10;

        start = 1'b0;


        // --------------------------------------------------------
        // Wait for complete network
        // --------------------------------------------------------

        wait(done == 1'b1);

        #1;


        // --------------------------------------------------------
        // Display result
        // --------------------------------------------------------

        $display("");
        $display("================================================");
        $display("COMPLETE ANN FINISHED");
        $display("================================================");
        $display("");

        $display(
            "Vivado / Verilog prediction = %0d",
            predicted_digit
        );

        $display("");


        // --------------------------------------------------------
        // Automatically verify prediction
        // --------------------------------------------------------

        if (predicted_digit == 4'd7) begin

            $display("************************************************");
            $display("REAL MNIST TEST PASSED");
            $display("************************************************");
            $display("");

            $display("Actual MNIST digit     : 7");
            $display("Python prediction      : 7");
            $display("Q1.15 reference        : 7");
            $display(
                "Verilog ANN prediction : %0d",
                predicted_digit
            );

            $display("");
            $display("ALL RESULTS MATCH.");

        end

        else begin

            $display("************************************************");
            $display("REAL MNIST TEST FAILED");
            $display("************************************************");
            $display("");

            $display("Expected digit         : 7");

            $display(
                "Verilog ANN prediction : %0d",
                predicted_digit
            );

        end


        // --------------------------------------------------------
        // Show complete hardware path
        // --------------------------------------------------------

        $display("");
        $display("Complete hardware path:");
        $display("");

        $display("Real 28 x 28 MNIST image");
        $display("          |");
        $display("          v");
        $display("784 Q1.15 pixel values");
        $display("          |");
        $display("          v");
        $display("32 hidden neurons");
        $display("          |");
        $display("        ReLU");
        $display("          |");
        $display("          v");
        $display("10 output neurons");
        $display("          |");
        $display("          v");
        $display("        Argmax");
        $display("          |");
        $display("          v");

        $display(
            "Predicted digit = %0d",
            predicted_digit
        );

        $display("");
        $display("================================================");
        $display("REAL MNIST FPGA ANN TEST COMPLETED");
        $display("================================================");
        $display("");


        #20;

        $finish;

    end


    // ============================================================
    // Safety timeout
    //
    // If something goes wrong and done never arrives,
    // terminate simulation instead of waiting forever.
    // ============================================================

    initial begin

        #1000000;

        $display("");
        $display("================================================");
        $display("ERROR: REAL MNIST ANN TEST TIMED OUT");
        $display("The network never asserted DONE.");
        $display("================================================");
        $display("");

        $finish;

    end


endmodule