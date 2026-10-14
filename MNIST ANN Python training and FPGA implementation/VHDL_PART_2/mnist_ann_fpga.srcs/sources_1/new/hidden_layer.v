module hidden_layer (
    input  wire clk,
    input  wire reset,
    input  wire start,

    input  wire signed [15:0] pixel_in,

    input  wire [4:0] read_index,
    output wire signed [15:0] hidden_out,

    output reg  [9:0] pixel_index,
    output reg  [4:0] neuron_index,
    output reg  done
);

    // Stores outputs of all 32 hidden neurons
    reg signed [15:0] hidden_outputs [0:31];

    // Arithmetic registers
    reg signed [47:0] accumulator;
    reg signed [31:0] product;
    reg signed [47:0] final_sum;

    reg running;

    // Current weight and bias
    wire signed [15:0] weight_value;
    reg  signed [15:0] bias_value;

    // Weight memory address
    wire [14:0] weight_address;

    // Properly aligned 48-bit bias
    wire signed [47:0] bias_aligned;

    // Allows the next layer to read any hidden neuron output
    assign hidden_out = hidden_outputs[read_index];

    // Address = neuron number * 784 + pixel number
    assign weight_address =
        (neuron_index * 15'd784) + pixel_index;

    // Sign-extend bias from 16 bits to 48 bits,
    // then shift left by 15 to match Q2.30 accumulator scale
    assign bias_aligned =
        {{32{bias_value[15]}}, bias_value} <<< 15;

    // Full hidden-layer weight memory
    hidden_weights_mem weight_memory (
        .addr(weight_address),
        .weight_out(weight_value)
    );

    // Hidden-layer biases
    always @(*) begin
        case (neuron_index)

            5'd0:  bias_value = 16'sd5339;
            5'd1:  bias_value = 16'sd2977;
            5'd2:  bias_value = 16'sd477;
            5'd3:  bias_value = 16'sd952;
            5'd4:  bias_value = 16'sd9911;
            5'd5:  bias_value = 16'sd2649;
            5'd6:  bias_value = -16'sd7725;
            5'd7:  bias_value = 16'sd10102;

            5'd8:  bias_value = 16'sd4262;
            5'd9:  bias_value = 16'sd3090;
            5'd10: bias_value = 16'sd1936;
            5'd11: bias_value = -16'sd763;
            5'd12: bias_value = 16'sd7956;
            5'd13: bias_value = 16'sd6282;
            5'd14: bias_value = -16'sd7206;
            5'd15: bias_value = -16'sd8579;

            5'd16: bias_value = 16'sd4951;
            5'd17: bias_value = 16'sd13971;
            5'd18: bias_value = 16'sd7074;
            5'd19: bias_value = 16'sd1373;
            5'd20: bias_value = 16'sd4782;
            5'd21: bias_value = 16'sd763;
            5'd22: bias_value = -16'sd737;
            5'd23: bias_value = -16'sd5893;

            5'd24: bias_value = 16'sd6816;
            5'd25: bias_value = -16'sd3077;
            5'd26: bias_value = -16'sd1200;
            5'd27: bias_value = 16'sd7144;
            5'd28: bias_value = 16'sd4333;
            5'd29: bias_value = -16'sd641;
            5'd30: bias_value = 16'sd5016;
            5'd31: bias_value = 16'sd3028;

            default: bias_value = 16'sd0;

        endcase
    end

    // Hidden-layer controller
    always @(posedge clk) begin

        if (reset) begin

            pixel_index  <= 10'd0;
            neuron_index <= 5'd0;

            accumulator  <= 48'sd0;
            product      <= 32'sd0;
            final_sum    <= 48'sd0;

            running      <= 1'b0;
            done         <= 1'b0;

        end

        else begin

            // done only pulses for one clock cycle
            done <= 1'b0;

            // Start calculation of full hidden layer
            if (start && !running) begin

                pixel_index  <= 10'd0;
                neuron_index <= 5'd0;

                accumulator  <= 48'sd0;

                running      <= 1'b1;

            end

            else if (running) begin

                // Q1.15 x Q1.15 = Q2.30
                product = pixel_in * weight_value;

                // Process pixels 0 to 782
                if (pixel_index < 10'd783) begin

                    accumulator <= accumulator + product;

                    pixel_index <= pixel_index + 1'b1;

                end

                // Pixel 783 is the final input for the current neuron
                else begin

                    // Add:
                    // accumulated products
                    // + final product
                    // + aligned bias
                    final_sum =
                        accumulator +
                        product +
                        bias_aligned;

                    // ReLU
                    if (final_sum <= 0) begin

                        hidden_outputs[neuron_index] <= 16'sd0;

                    end

                    // Positive saturation
                    else if ((final_sum >>> 15) > 32767) begin

                        hidden_outputs[neuron_index] <= 16'sd32767;

                    end

                    // Convert Q2.30 result back to Q1.15
                    else begin

                        hidden_outputs[neuron_index]
                            <= final_sum >>> 15;

                    end

                    // If neuron 31 is finished,
                    // the whole hidden layer is complete
                    if (neuron_index == 5'd31) begin

                        running <= 1'b0;
                        done    <= 1'b1;

                    end

                    // Otherwise move to the next hidden neuron
                    else begin

                        neuron_index <= neuron_index + 1'b1;
                        pixel_index  <= 10'd0;

                        accumulator  <= 48'sd0;

                    end
                end
            end
        end
    end

endmodule