module output_layer (
    input  wire clk,
    input  wire reset,
    input  wire start,

    // Hidden-layer value currently selected
    input  wire signed [15:0] hidden_in,

    // Tells hidden_layer which of its 32 outputs we need
    output reg [4:0] hidden_read_index,

    // Allows another module (later argmax) to read
    // any of the 10 completed output scores
    input  wire [3:0] read_index,
    output wire signed [47:0] score_out,

    // Current output neuron being calculated
    output reg [3:0] neuron_index,

    output reg done
);

    // ------------------------------------------------------------
    // Store the 10 output neuron scores
    //
    // We keep these at 48-bit Q2.30-style precision instead of
    // reducing them to 16 bits.
    //
    // This is useful because argmax only needs to know which score
    // is largest, and keeping full precision avoids unnecessary
    // clipping or overflow.
    // ------------------------------------------------------------

    reg signed [47:0] output_scores [0:9];


    // ------------------------------------------------------------
    // Arithmetic registers
    // ------------------------------------------------------------

    reg signed [47:0] accumulator;
    reg signed [31:0] product;
    reg signed [47:0] final_sum;

    reg running;


    // ------------------------------------------------------------
    // Weight and bias signals
    // ------------------------------------------------------------

    wire signed [15:0] weight_value;
    reg  signed [15:0] bias_value;

    wire signed [47:0] bias_aligned;

    wire [8:0] weight_address;


    // ------------------------------------------------------------
    // Read completed output scores
    // ------------------------------------------------------------

    assign score_out = output_scores[read_index];


    // ------------------------------------------------------------
    // Weight addressing
    //
    // 10 output neurons
    // 32 hidden inputs per neuron
    //
    // Neuron 0 -> addresses   0 to 31
    // Neuron 1 -> addresses  32 to 63
    // ...
    // Neuron 9 -> addresses 288 to 319
    // ------------------------------------------------------------

    assign weight_address =
        (neuron_index * 9'd32) + hidden_read_index;


    // ------------------------------------------------------------
    // Bias alignment
    //
    // Bias is Q1.15.
    // Products are Q2.30.
    //
    // Sign-extend bias to 48 bits and shift left by 15.
    // ------------------------------------------------------------

    assign bias_aligned =
        {{32{bias_value[15]}}, bias_value} <<< 15;


    // ------------------------------------------------------------
    // Output-layer weight memory
    // ------------------------------------------------------------

    output_weights_mem weight_memory (
        .addr(weight_address),
        .weight_out(weight_value)
    );


    // ------------------------------------------------------------
    // Output-layer biases
    //
    // Actual values from output_bias_q15.txt
    // ------------------------------------------------------------

    always @(*) begin

        case (neuron_index)

            4'd0: bias_value = -16'sd5259;
            4'd1: bias_value =  16'sd1166;
            4'd2: bias_value =  16'sd2711;
            4'd3: bias_value =  16'sd453;
            4'd4: bias_value =  16'sd10107;
            4'd5: bias_value =  16'sd1791;
            4'd6: bias_value =  16'sd4674;
            4'd7: bias_value =  16'sd4038;
            4'd8: bias_value = -16'sd2848;
            4'd9: bias_value = -16'sd4430;

            default:
                bias_value = 16'sd0;

        endcase

    end


    // ------------------------------------------------------------
    // Output-layer controller
    // ------------------------------------------------------------

    always @(posedge clk) begin

        if (reset) begin

            hidden_read_index <= 5'd0;
            neuron_index      <= 4'd0;

            accumulator       <= 48'sd0;
            product           <= 32'sd0;
            final_sum         <= 48'sd0;

            running           <= 1'b0;
            done              <= 1'b0;

        end

        else begin

            // done is a one-clock pulse
            done <= 1'b0;


            // ----------------------------------------------------
            // Start complete 32 -> 10 output layer
            // ----------------------------------------------------

            if (start && !running) begin

                hidden_read_index <= 5'd0;
                neuron_index      <= 4'd0;

                accumulator       <= 48'sd0;

                running           <= 1'b1;

            end


            // ----------------------------------------------------
            // Perform calculations
            // ----------------------------------------------------

            else if (running) begin

                // Q1.15 hidden value x Q1.15 weight
                // = Q2.30 product
                product = hidden_in * weight_value;


                // ------------------------------------------------
                // Hidden inputs 0 to 30
                // ------------------------------------------------

                if (hidden_read_index < 5'd31) begin

                    accumulator <= accumulator + product;

                    hidden_read_index
                        <= hidden_read_index + 1'b1;

                end


                // ------------------------------------------------
                // Hidden input 31 = final input for this
                // output neuron
                // ------------------------------------------------

                else begin

                    final_sum =
                        accumulator +
                        product +
                        bias_aligned;


                    // No ReLU here.
                    // Store the signed output score directly.

                    output_scores[neuron_index]
                        <= final_sum;


                    // ------------------------------------------------
                    // Output neuron 9 finished:
                    // entire output layer is complete
                    // ------------------------------------------------

                    if (neuron_index == 4'd9) begin

                        running <= 1'b0;
                        done    <= 1'b1;

                    end


                    // ------------------------------------------------
                    // Otherwise begin next output neuron
                    // ------------------------------------------------

                    else begin

                        neuron_index
                            <= neuron_index + 1'b1;

                        hidden_read_index
                            <= 5'd0;

                        accumulator
                            <= 48'sd0;

                    end

                end

            end

        end

    end

endmodule