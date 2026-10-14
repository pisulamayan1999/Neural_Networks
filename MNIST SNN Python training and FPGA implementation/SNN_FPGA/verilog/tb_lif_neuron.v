`timescale 1ns / 1ps

module tb_lif_neuron;

    localparam integer TEST_STEPS = 32;

    reg clk;
    reg reset;
    reg enable;
    reg signed [31:0] current_in;

    wire signed [31:0] membrane_out;
    wire spike_out;
    wire valid_out;

    reg signed [31:0] test_currents [0:TEST_STEPS-1];
    reg signed [31:0] expected_membranes [0:TEST_STEPS-1];
    reg        [7:0]  expected_spikes [0:TEST_STEPS-1];

    reg signed [31:0] held_membrane;

    integer index;
    integer pass_number;
    integer failures;
    integer checks;

    lif_neuron dut (
        .clk          (clk),
        .reset        (reset),
        .enable       (enable),
        .current_in   (current_in),
        .membrane_out (membrane_out),
        .spike_out    (spike_out),
        .valid_out    (valid_out)
    );

    // 40 ns period = 25 MHz.
    initial begin
        clk = 1'b0;
        forever #20 clk = ~clk;
    end

    initial begin
        reset     = 1'b1;
        enable    = 1'b0;
        current_in = 32'sd0;

        failures = 0;
        checks   = 0;

        // Initialize to unknown so missing/incomplete files
        // are detected explicitly.
        for (index = 0; index < TEST_STEPS; index = index + 1) begin
            test_currents[index]      = 32'hxxxxxxxx;
            expected_membranes[index] = 32'hxxxxxxxx;
            expected_spikes[index]    = 8'hxx;
        end

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/lif_test_currents.mem",
            test_currents
        );

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/lif_test_membranes.mem",
            expected_membranes
        );

        $readmemh(
            "D:/Internship_Projects/Step_2_SNN/SNN_FPGA/fpga_exports/lif_test_spikes.mem",
            expected_spikes
        );

        for (index = 0; index < TEST_STEPS; index = index + 1) begin
            if (
                (^test_currents[index] === 1'bx)
                || (^expected_membranes[index] === 1'bx)
                || (^expected_spikes[index] === 1'bx)
            ) begin
                $fatal(
                    1,
                    "Missing or invalid test vector at address %0d. Check .mem paths.",
                    index
                );
            end

            if (
                expected_spikes[index] !== 8'd0
                && expected_spikes[index] !== 8'd1
            ) begin
                $fatal(1, "Invalid spike reference at address %0d", index);
            end
        end

        $display("==============================================");
        $display("LIF NEURON: VERILOG VS PYTHON INTEGER REFERENCE");
        $display("==============================================");

        // Run the entire sequence twice, with reset between
        // passes, to verify repeatable initialization.
        for (
            pass_number = 1;
            pass_number <= 2;
            pass_number = pass_number + 1
        ) begin

            @(negedge clk);
            reset  = 1'b1;
            enable = 1'b0;

            @(posedge clk);
            #1;

            if (
                membrane_out !== 32'sd0
                || spike_out !== 1'b0
                || valid_out !== 1'b0
            ) begin
                $fatal(1, "Reset failed.");
            end

            @(negedge clk);
            reset = 1'b0;

            for (
                index = 0;
                index < TEST_STEPS;
                index = index + 1
            ) begin

                // Drive inputs away from the sampling edge.
                @(negedge clk);
                enable     = 1'b1;
                current_in = test_currents[index];

                @(posedge clk);
                #1;

                checks = checks + 1;

                if (
                    membrane_out !== expected_membranes[index]
                    || spike_out !== expected_spikes[index][0]
                    || valid_out !== 1'b1
                ) begin
                    failures = failures + 1;

                    $display(
                        "FAIL pass=%0d step=%0d current=%0d membrane=%0d expected=%0d spike=%b expected=%b valid=%b",
                        pass_number,
                        index + 1,
                        $signed(current_in),
                        $signed(membrane_out),
                        $signed(expected_membranes[index]),
                        spike_out,
                        expected_spikes[index][0],
                        valid_out
                    );
                end
                else begin
                    $display(
                        "PASS pass=%0d step=%0d membrane=%0d spike=%b",
                        pass_number,
                        index + 1,
                        $signed(membrane_out),
                        spike_out
                    );
                end

                // Insert a disabled clock between time steps.
                // An unaccepted current must not change state.
                held_membrane = membrane_out;

                @(negedge clk);
                enable     = 1'b0;
                current_in = 32'sd123456;

                @(posedge clk);
                #1;

                if (
                    membrane_out !== held_membrane
                    || spike_out !== 1'b0
                    || valid_out !== 1'b0
                ) begin
                    failures = failures + 1;
                    $display(
                        "FAIL: disabled state changed at pass=%0d step=%0d",
                        pass_number,
                        index + 1
                    );
                end
            end
        end

        $display("==============================================");
        $display("Reference steps checked: %0d", checks);
        $display("Failures: %0d", failures);

        if (failures == 0) begin
            $display("ALL TESTS PASSED");
            $display("Both passes matched Python exactly.");
            $display("Reset and disabled-state checks passed.");
        end
        else begin
            $fatal(1, "LIF verification failed.");
        end

        $display("==============================================");
        $finish;
    end

endmodule