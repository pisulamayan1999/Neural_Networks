# ============================================================
# Genesys ZU-5EV Rev. D
# MNIST ANN FPGA Hardware Test
# ============================================================


# ------------------------------------------------------------
# 25 MHz PL system clock
# ------------------------------------------------------------

set_property -dict { PACKAGE_PIN E12 IOSTANDARD LVCMOS18 } [get_ports { clk }]

create_clock -period 40.000 -name sys_clk [get_ports { clk }]


# ------------------------------------------------------------
# Reset button
#
# BTN2
# ------------------------------------------------------------

set_property -dict { PACKAGE_PIN B10 IOSTANDARD LVCMOS18 } [get_ports { reset }]


# ------------------------------------------------------------
# Start button
#
# BTN3
# ------------------------------------------------------------

set_property -dict { PACKAGE_PIN H12 IOSTANDARD LVCMOS18 } [get_ports { start }]


# ------------------------------------------------------------
# Prediction LEDs
#
# predicted_digit[0] -> led0
# predicted_digit[1] -> led1
# predicted_digit[2] -> led2
# predicted_digit[3] -> led3
# ------------------------------------------------------------

set_property -dict { PACKAGE_PIN L14 IOSTANDARD LVCMOS33 } [get_ports { led0 }]

set_property -dict { PACKAGE_PIN L13 IOSTANDARD LVCMOS33 } [get_ports { led1 }]

set_property -dict { PACKAGE_PIN K14 IOSTANDARD LVCMOS33 } [get_ports { led2 }]

set_property -dict { PACKAGE_PIN J14 IOSTANDARD LVCMOS33 } [get_ports { led3 }]


# ------------------------------------------------------------
# ANN finished indicator
#
# Green channel of RGB LED LD5
# ------------------------------------------------------------

set_property -dict { PACKAGE_PIN B9 IOSTANDARD LVCMOS12 } [get_ports { led_done }]