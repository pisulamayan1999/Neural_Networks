`timescale 1ns / 1ps

module tb_hidden_layer;

    reg clk;
    reg reset;
    reg start;

    reg signed [15:0] pixel_in;

    reg [4:0] read_index;

    wire signed [15:0] hidden_out;

    wire [9:0] pixel_index;
    wire [4:0] neuron_index;
    wire done;

    integer i;
    integer errors;


    // ------------------------------------------------------------
    // Device Under Test
    // ------------------------------------------------------------

    hidden_layer uut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .pixel_in(pixel_in),

        .read_index(read_index),
        .hidden_out(hidden_out),

        .pixel_index(pixel_index),
        .neuron_index(neuron_index),

        .done(done)
    );


    // ------------------------------------------------------------
    // Clock
    // 10 ns period = 100 MHz
    // ------------------------------------------------------------

    initial begin
        clk = 1'b0;

        forever begin
            #5 clk = ~clk;
        end
    end


    // ------------------------------------------------------------
    // Pixel input
    //
    // pixel 0 = 16384 = 0.5 in Q1.15
    // pixels 1 to 783 = 0
    //
    // The hidden layer provides pixel_index,
    // so the testbench changes pixel_in based on that index.
    // ------------------------------------------------------------

    always @(*) begin

        if (pixel_index == 10'd0)
            pixel_in = 16'sd16384;

        else
            pixel_in = 16'sd0;

    end


    // ------------------------------------------------------------
    // Expected outputs
    //
    // output = ReLU(0.5 * first_weight + bias)
    //
    // These values were calculated using the same Q1.15 arithmetic
    // used by the Verilog hidden layer.
    // ------------------------------------------------------------

    function signed [15:0] expected_output;

        input [4:0] index;

        begin

            case (index)

                5'd0:  expected_output = 16'sd5884;
                5'd1:  expected_output = 16'sd3467;
                5'd2:  expected_output = 16'sd537;
                5'd3:  expected_output = 16'sd530;
                5'd4:  expected_output = 16'sd10319;
                5'd5:  expected_output = 16'sd2265;

                5'd6:  expected_output = 16'sd0;

                5'd7:  expected_output = 16'sd10298;
                5'd8:  expected_output = 16'sd3998;
                5'd9:  expected_output = 16'sd3248;
                5'd10: expected_output = 16'sd1596;

                5'd11: expected_output = 16'sd0;

                5'd12: expected_output = 16'sd8275;
                5'd13: expected_output = 16'sd6436;

                5'd14: expected_output = 16'sd0;
                5'd15: expected_output = 16'sd0;

                5'd16: expected_output = 16'sd4521;
                5'd17: expected_output = 16'sd14227;
                5'd18: expected_output = 16'sd7044;
                5'd19: expected_output = 16'sd1140;
                5'd20: expected_output = 16'sd4923;
                5'd21: expected_output = 16'sd806;

                5'd22: expected_output = 16'sd0;
                5'd23: expected_output = 16'sd0;

                5'd24: expected_output = 16'sd6365;

                5'd25: expected_output = 16'sd0;
                5'd26: expected_output = 16'sd0;

                5'd27: expected_output = 16'sd7279;
                5'd28: expected_output = 16'sd4036;

                5'd29: expected_output = 16'sd0;

                5'd30: expected_output = 16'sd5159;
                5'd31: expected_output = 16'sd3551;

                default:
                    expected_output = 16'sd0;

            endcase

        end

    endfunction


    // ------------------------------------------------------------
    // Main test
    // ------------------------------------------------------------

    initial begin

        reset      = 1'b1;
        start      = 1'b0;
        read_index = 5'd0;

        errors = 0;


        // Hold reset
        #30;

        reset = 1'b0;

        #20;


        // --------------------------------------------------------
        // Start hidden-layer calculation
        // --------------------------------------------------------

        $display("");
        $display("==========================================");
        $display("Starting hidden layer NON-ZERO input test");
        $display("Pixel 0 = 16384 (0.5 Q1.15)");
        $display("Pixels 1-783 = 0");
        $display("==========================================");
        $display("");


        start = 1'b1;

        #10;

        start = 1'b0;


        // --------------------------------------------------------
        // Wait until all 32 neurons are finished
        // --------------------------------------------------------

        wait(done == 1'b1);

        #10;


        $display("");
        $display("Hidden layer finished.");
        $display("Checking all 32 neuron outputs...");
        $display("");


        // --------------------------------------------------------
        // Verify all 32 hidden outputs
        // --------------------------------------------------------

        for (i = 0; i < 32; i = i + 1) begin

            read_index = i;

            #1;


            if (hidden_out === expected_output(i)) begin

                $display(
                    "Neuron %0d PASSED | Output = %0d | Expected = %0d",
                    i,
                    hidden_out,
                    expected_output(i)
                );

            end

            else begin

                $display(
                    "Neuron %0d FAILED | Output = %0d | Expected = %0d",
                    i,
                    hidden_out,
                    expected_output(i)
                );

                errors = errors + 1;

            end

        end


        // --------------------------------------------------------
        // Final result
        // --------------------------------------------------------

        $display("");
        $display("==========================================");

        if (errors == 0) begin

            $display("ALL 32 NON-ZERO HIDDEN NEURON TESTS PASSED");

        end

        else begin

            $display("%0d HIDDEN NEURON TESTS FAILED", errors);

        end

        $display("==========================================");
        $display("");


        #20;

        $finish;

    end


endmodule