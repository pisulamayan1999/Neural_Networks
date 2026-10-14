import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms


# ============================================================
# Neural network definition
# Must match the trained model
# ============================================================

class TinyANN(nn.Module):
    def __init__(self):
        super(TinyANN, self).__init__()

        self.hidden = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):

        x = x.view(-1, 784)

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x


# ============================================================
# Helper functions
# ============================================================

def load_matrix(filename):

    matrix = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if line:
                matrix.append(
                    [int(value) for value in line.split()]
                )

    return matrix


def load_vector(filename):

    with open(filename, "r") as f:

        values = [
            int(value)
            for value in f.read().split()
        ]

    return values


def pixel_to_q15(value):

    q = int(round(value * 32768.0))

    # Q1.15 signed maximum
    if q > 32767:
        q = 32767

    if q < -32768:
        q = -32768

    return q


def to_hex16(value):

    return f"{value & 0xFFFF:04X}"


# ============================================================
# Fixed-point FPGA-style inference
#
# This intentionally follows the arithmetic used in the
# Verilog implementation.
# ============================================================

def fpga_inference(
    pixels_q15,
    hidden_weights,
    hidden_bias,
    output_weights,
    output_bias
):

    hidden_outputs = []


    # --------------------------------------------------------
    # Hidden layer
    # 784 -> 32
    # --------------------------------------------------------

    for neuron in range(32):

        accumulator = 0

        for pixel_index in range(784):

            product = (
                pixels_q15[pixel_index]
                * hidden_weights[neuron][pixel_index]
            )

            accumulator += product


        # Bias must be aligned from Q1.15 to Q2.30
        final_sum = (
            accumulator
            + (hidden_bias[neuron] << 15)
        )


        # ReLU
        if final_sum <= 0:

            hidden_value = 0

        else:

            hidden_value = final_sum >> 15

            # Same positive saturation used in Verilog
            if hidden_value > 32767:
                hidden_value = 32767


        hidden_outputs.append(hidden_value)


    # --------------------------------------------------------
    # Output layer
    # 32 -> 10
    #
    # No ReLU here.
    # Keep full Q2.30-style scores.
    # --------------------------------------------------------

    output_scores = []

    for neuron in range(10):

        accumulator = 0

        for hidden_index in range(32):

            product = (
                hidden_outputs[hidden_index]
                * output_weights[neuron][hidden_index]
            )

            accumulator += product


        score = (
            accumulator
            + (output_bias[neuron] << 15)
        )

        output_scores.append(score)


    # --------------------------------------------------------
    # Argmax
    # --------------------------------------------------------

    predicted_digit = max(
        range(10),
        key=lambda i: output_scores[i]
    )


    return hidden_outputs, output_scores, predicted_digit


# ============================================================
# Main program
# ============================================================

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# Load trained PyTorch model
# ------------------------------------------------------------

model_path = os.path.join(
    script_folder,
    "tiny_ann_mnist.pth"
)

model = TinyANN()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()


# ------------------------------------------------------------
# Load Q1.15 parameters used by FPGA
# ------------------------------------------------------------

hidden_weights = load_matrix(
    os.path.join(
        script_folder,
        "hidden_weight_q15.txt"
    )
)

hidden_bias = load_vector(
    os.path.join(
        script_folder,
        "hidden_bias_q15.txt"
    )
)

output_weights = load_matrix(
    os.path.join(
        script_folder,
        "output_weight_q15.txt"
    )
)

output_bias = load_vector(
    os.path.join(
        script_folder,
        "output_bias_q15.txt"
    )
)


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

assert len(hidden_weights) == 32
assert all(len(row) == 784 for row in hidden_weights)

assert len(hidden_bias) == 32

assert len(output_weights) == 10
assert all(len(row) == 32 for row in output_weights)

assert len(output_bias) == 10


# ------------------------------------------------------------
# Load real MNIST test dataset
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(
    root=os.path.join(script_folder, "data"),
    train=False,
    download=True,
    transform=transform
)


# ------------------------------------------------------------
# Find a clean test example
#
# Requirements:
#
# true label
#     =
# floating-point Python prediction
#     =
# Q1.15 FPGA-style prediction
# ------------------------------------------------------------

chosen = None


for test_index in range(len(test_dataset)):

    image, label = test_dataset[test_index]


    # --------------------------------------------------------
    # Normal Python model
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image.unsqueeze(0)
        )

        python_prediction = int(
            torch.argmax(
                logits,
                dim=1
            ).item()
        )


    # --------------------------------------------------------
    # Convert image to Q1.15
    # --------------------------------------------------------

    flat_pixels = image.view(-1).tolist()

    pixels_q15 = [
        pixel_to_q15(value)
        for value in flat_pixels
    ]


    # --------------------------------------------------------
    # Hardware-style prediction
    # --------------------------------------------------------

    (
        hidden_outputs,
        output_scores,
        fpga_prediction
    ) = fpga_inference(
        pixels_q15,
        hidden_weights,
        hidden_bias,
        output_weights,
        output_bias
    )


    # Pick first example where everything agrees
    if (
        python_prediction == label
        and
        fpga_prediction == label
    ):

        chosen = {
            "index": test_index,
            "label": int(label),
            "python_prediction": python_prediction,
            "fpga_prediction": fpga_prediction,
            "pixels_q15": pixels_q15,
            "hidden_outputs": hidden_outputs,
            "output_scores": output_scores
        }

        break


# ------------------------------------------------------------
# Make sure we found one
# ------------------------------------------------------------

if chosen is None:

    raise RuntimeError(
        "Could not find a matching MNIST test image."
    )


# ============================================================
# Write image memory file for Vivado
# ============================================================

mem_path = os.path.join(
    script_folder,
    "mnist_image.mem"
)

with open(mem_path, "w") as f:

    for pixel in chosen["pixels_q15"]:

        f.write(
            to_hex16(pixel)
            + "\n"
        )


# ============================================================
# Write expected information
# ============================================================

info_path = os.path.join(
    script_folder,
    "mnist_test_expected.txt"
)

with open(info_path, "w") as f:

    f.write(
        f"MNIST test index: "
        f"{chosen['index']}\n"
    )

    f.write(
        f"True digit: "
        f"{chosen['label']}\n"
    )

    f.write(
        f"Python prediction: "
        f"{chosen['python_prediction']}\n"
    )

    f.write(
        f"Q1.15 FPGA-style prediction: "
        f"{chosen['fpga_prediction']}\n"
    )

    f.write("\nOutput scores:\n")

    for i, score in enumerate(
        chosen["output_scores"]
    ):

        f.write(
            f"Digit {i}: {score}\n"
        )


# ============================================================
# Console output
# ============================================================

print("")
print("============================================")
print("REAL MNIST FPGA TEST IMAGE CREATED")
print("============================================")
print("")

print(
    "MNIST test index:",
    chosen["index"]
)

print(
    "True digit:",
    chosen["label"]
)

print(
    "Python prediction:",
    chosen["python_prediction"]
)

print(
    "Q1.15 FPGA-style prediction:",
    chosen["fpga_prediction"]
)

print("")

print(
    "Created:",
    mem_path
)

print(
    "Created:",
    info_path
)

print("")

print(
    "Number of pixels written:",
    len(chosen["pixels_q15"])
)

print("")

print("Output scores:")

for i, score in enumerate(
    chosen["output_scores"]
):

    print(
        f"Digit {i}: {score}"
    )

print("")
print("============================================")
print(
    "NEXT: LOAD mnist_image.mem INTO VIVADO"
)
print("============================================")