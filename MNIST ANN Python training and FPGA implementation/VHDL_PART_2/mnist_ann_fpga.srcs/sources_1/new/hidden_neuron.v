module hidden_neuron (
    input  wire clk,
    input  wire reset,
    input  wire start,

    input  wire signed [15:0] pixel_in,
    input  wire signed [15:0] weight_in,
    input  wire signed [15:0] bias_in,

    output reg  [9:0] pixel_index,
    output reg  done,
    output reg  signed [15:0] neuron_out
);

    reg signed [31:0] product;
    reg signed [47:0] accumulator;

    reg running;

    reg signed [47:0] final_sum;

    always @(posedge clk) begin
        if (reset) begin
            pixel_index <= 10'd0;
            product     <= 32'sd0;
            accumulator <= 48'sd0;
            neuron_out  <= 16'sd0;
            done        <= 1'b0;
            running     <= 1'b0;
            final_sum   <= 48'sd0;
        end

        else begin
            done <= 1'b0;

            if (start && !running) begin
                pixel_index <= 10'd0;
                accumulator <= 48'sd0;
                neuron_out  <= 16'sd0;
                running     <= 1'b1;
            end

            else if (running) begin
                product = pixel_in * weight_in;

                if (pixel_index < 10'd783) begin
                    accumulator <= accumulator + product;
                    pixel_index <= pixel_index + 1'b1;
                end

                else begin
                    final_sum =
                        accumulator
                        + product
                        + ($signed(bias_in) <<< 15);

                    if (final_sum <= 0) begin
                        neuron_out <= 16'sd0;
                    end

                    else if ((final_sum >>> 15) > 32767) begin
                        neuron_out <= 16'sd32767;
                    end

                    else begin
                        neuron_out <= final_sum >>> 15;
                    end

                    running <= 1'b0;
                    done    <= 1'b1;
                end
            end
        end
    end

endmodule