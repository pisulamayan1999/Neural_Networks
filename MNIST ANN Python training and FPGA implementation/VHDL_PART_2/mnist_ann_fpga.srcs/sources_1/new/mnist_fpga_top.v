module mnist_fpga_top (
    input  wire clk,
    input  wire reset,
    input  wire start,

    output wire led0,
    output wire led1,
    output wire led2,
    output wire led3,
    output wire led_done
);

    // ------------------------------------------------------------
    // Real MNIST image memory
    // ------------------------------------------------------------

    reg signed [15:0] image_memory [0:783];

    initial begin
        $readmemh("mnist_image.mem", image_memory);
    end


    // ------------------------------------------------------------
    // ANN signals
    // ------------------------------------------------------------

    wire [9:0] pixel_index;

    wire signed [15:0] pixel_value;

    wire [3:0] predicted_digit;

    wire ann_done;


    // ------------------------------------------------------------
    // Start-button edge detection
    //
    // Converts a long physical button press into one clock pulse.
    // ------------------------------------------------------------

    reg start_previous;

    wire start_pulse;

    assign start_pulse = start & ~start_previous;


    always @(posedge clk) begin

        if (reset)
            start_previous <= 1'b0;

        else
            start_previous <= start;

    end


    // ------------------------------------------------------------
    // Latch DONE so that the LED remains visible
    //
    // Cleared by reset or when a new ANN calculation starts.
    // ------------------------------------------------------------

    reg done_latched;

    always @(posedge clk) begin

        if (reset)
            done_latched <= 1'b0;

        else if (start_pulse)
            done_latched <= 1'b0;

        else if (ann_done)
            done_latched <= 1'b1;

    end


    // ------------------------------------------------------------
    // Supply requested pixel to ANN
    // ------------------------------------------------------------

    assign pixel_value = image_memory[pixel_index];


    // ------------------------------------------------------------
    // Complete ANN
    // ------------------------------------------------------------

    mnist_ann ann_inst (
        .clk(clk),
        .reset(reset),
        .start(start_pulse),

        .pixel_in(pixel_value),

        .pixel_index(pixel_index),

        .predicted_digit(predicted_digit),

        .done(ann_done)
    );


    // ------------------------------------------------------------
    // Predicted digit LEDs
    //
    // Binary:
    //
    // 0 = 0000
    // 1 = 0001
    // 2 = 0010
    // ...
    // 7 = 0111
    // 9 = 1001
    // ------------------------------------------------------------

    assign led0 = predicted_digit[0];
    assign led1 = predicted_digit[1];
    assign led2 = predicted_digit[2];
    assign led3 = predicted_digit[3];


    // ------------------------------------------------------------
    // Finished indicator
    // ------------------------------------------------------------

    assign led_done = done_latched;


endmodule