import torch
import snntorch as snn


# ------------------------------------------------------------
# WEIGHTS
# ------------------------------------------------------------

weights = torch.tensor([
    0.6,
   -0.4,
    0.7
])

bias = 0.1


# ------------------------------------------------------------
# INPUT SPIKES OVER 10 TIME STEPS
#
# Each row = one time step
#
# 1 = neuron fired
# 0 = neuron did not fire
# ------------------------------------------------------------

input_spikes = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0]
])


# ------------------------------------------------------------
# CREATE ONE LIF NEURON
# ------------------------------------------------------------

beta = 0.9
threshold = 1.0

lif = snn.Leaky(
    beta=beta,
    threshold=threshold
)

mem = torch.tensor([0.0])


print("Spiking input test")
print("--------------------------------")


# ------------------------------------------------------------
# RUN THROUGH TIME
# ------------------------------------------------------------

for step in range(len(input_spikes)):

    spikes = input_spikes[step]

    # Weighted sum of incoming spikes
    input_current = torch.sum(
        spikes * weights
    ) + bias

    # LIF neuron
    spike_out, mem = lif(
        input_current,
        mem
    )

    print(
        f"Time {step + 1}: "
        f"Inputs = {spikes.int().tolist()} | "
        f"Current = {input_current.item():.2f} | "
        f"Membrane = {mem.item():.4f} | "
        f"Output Spike = {int(spike_out.item())}"
    )