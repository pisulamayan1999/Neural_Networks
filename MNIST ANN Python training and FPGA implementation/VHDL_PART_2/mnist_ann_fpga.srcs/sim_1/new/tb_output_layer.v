`timescale 1ns / 1ps

module tb_output_layer;

    reg clk;
    reg reset;
    reg start;

    reg signed [15:0] hidden_in;

    wire [4:0] hidden_read_index;

    reg [3:0] read_index;
    wire signed [47:0] score_out;

    wire [3:0] neuron_index;
    wire done;

    integer i;
    integer errors;


    // ------------------------------------------------------------
    // Device Under Test
    // ------------------------------------------------------------

    output_layer uut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .hidden_in(hidden_in),

        .hidden_read_index(hidden_read_index),

        .read_index(read_index),
        .score_out(score_out),

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
    // Non-zero hidden input test
    //
    // hidden[0] = 16384 = 0.5 in Q1.15
    // hidden[1..31] = 0
    // ------------------------------------------------------------

    always @(*) begin

        if (hidden_read_index == 5'd0)
            hidden_in = 16'sd16384;

        else
            hidden_in = 16'sd0;

    end


    // ------------------------------------------------------------
    // Expected output scores
    //
    // score = (hidden[0] * weight[0]) + (bias << 15)
    //
    // Only hidden[0] is non-zero.
    // ------------------------------------------------------------

    function signed [47:0] expected_score;

        input [3:0] index;

        begin

            case (index)

                4'd0: expected_score = -48'sd192053248;
                4'd1: expected_score =  48'sd36864000;
                4'd2: expected_score =  48'sd256278528;
                4'd3: expected_score =  48'sd12959744;
                4'd4: expected_score = -48'sd20660224;
                4'd5: expected_score =  48'sd70139904;
                4'd6: expected_score =  48'sd284884992;
                4'd7: expected_score =  48'sd255787008;
                4'd8: expected_score = -48'sd377929728;
                4'd9: expected_score = -48'sd125206528;

                default:
                    expected_score = 48'sd0;

            endcase

        end

    endfunction


    // ------------------------------------------------------------
    // Main test
    // ------------------------------------------------------------

    initial begin

        reset      = 1'b1;
        start      = 1'b0;
        read_index = 4'd0;

        errors = 0;


        // Hold reset
        #30;

        reset = 1'b0;

        #20;


        // --------------------------------------------------------
        // Start output-layer calculation
        // --------------------------------------------------------

        $display("");
        $display("==========================================");
        $display("Starting output layer NON-ZERO input test");
        $display("Hidden[0] = 16384 (0.5 Q1.15)");
        $display("Hidden[1..31] = 0");
        $display("==========================================");
        $display("");


        start = 1'b1;

        #10;

        start = 1'b0;


        // --------------------------------------------------------
        // Wait until all 10 output neurons finish
        // --------------------------------------------------------

        wait(done == 1'b1);

        #10;


        $display("");
        $display("Output layer finished.");
        $display("Checking all 10 output scores...");
        $display("");


        // --------------------------------------------------------
        // Check all 10 output scores
        // --------------------------------------------------------

        for (i = 0; i < 10; i = i + 1) begin

            read_index = i;

            #1;


            if (score_out === expected_score(i)) begin

                $display(
                    "Output neuron %0d PASSED | Score = %0d | Expected = %0d",
                    i,
                    score_out,
                    expected_score(i)
                );

            end

            else begin

                $display(
                    "Output neuron %0d FAILED | Score = %0d | Expected = %0d",
                    i,
                    score_out,
                    expected_score(i)
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

            $display("ALL 10 NON-ZERO OUTPUT NEURONS PASSED");

        end

        else begin

            $display("%0d OUTPUT NEURON TESTS FAILED", errors);

        end

        $display("==========================================");
        $display("");


        #20;

        $finish;

    end

endmodule