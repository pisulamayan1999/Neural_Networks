import torch
import snntorch as snn


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

beta = 0.9
threshold = 1.0

# Create ONE Leaky Integrate-and-Fire neuron
lif = snn.Leaky(
    beta=beta,
    threshold=threshold
)

# Initial membrane potential
mem = torch.tensor([0.0])

# Constant input current
input_current = torch.tensor([0.4])


print("LIF neuron test")
print("----------------------------")
print("Beta:", beta)
print("Threshold:", threshold)
print("Input current:", input_current.item())
print()


# ------------------------------------------------------------
# RUN FOR 10 TIME STEPS
# ------------------------------------------------------------

for step in range(10):

    spike, mem = lif(
        input_current,
        mem
    )

    print(
        f"Time step {step + 1}: "
        f"Spike = {int(spike.item())}, "
        f"Membrane = {mem.item():.4f}"
    )