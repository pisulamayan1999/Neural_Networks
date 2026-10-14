import torch
import snntorch as snn


# ------------------------------------------------------------
# INPUT VALUES
# ------------------------------------------------------------

inputs = torch.tensor([
    0.5,
    0.2,
    0.9
])


# ------------------------------------------------------------
# WEIGHTS
# ------------------------------------------------------------

weights = torch.tensor([
    0.6,
   -0.4,
    0.7
])


# ------------------------------------------------------------
# BIAS
# ------------------------------------------------------------

bias = 0.1


# ------------------------------------------------------------
# CALCULATE WEIGHTED INPUT CURRENT
# ------------------------------------------------------------

input_current = (
    inputs[0] * weights[0]
    + inputs[1] * weights[1]
    + inputs[2] * weights[2]
    + bias
)


print("Weighted LIF neuron test")
print("------------------------")

print("Inputs :", inputs)
print("Weights:", weights)
print("Bias   :", bias)

print()
print("Calculated input current:", input_current.item())
print()


# ------------------------------------------------------------
# CREATE LIF NEURON
# ------------------------------------------------------------

beta = 0.9
threshold = 1.0

lif = snn.Leaky(
    beta=beta,
    threshold=threshold
)

mem = torch.tensor([0.0])


# ------------------------------------------------------------
# RUN THROUGH TIME
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