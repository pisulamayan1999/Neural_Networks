from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
CHECKPOINT = FOLDER / "snn_mnist_16_5epochs.pth"
EXPORT_FOLDER = FOLDER / "fpga_exports"

TEST_IMAGE_INDEX = 0

SCALE = 32768
ROUND_HALF = 16384


# ============================================================
# HELPERS
# ============================================================

def quantize(values):
    # Same rounding used in the validated fixed-point simulation.
    return np.rint(
        np.asarray(values, dtype=np.float64) * SCALE
    ).astype(np.int64)


def check_signed(values, bits, name):
    values = np.asarray(values, dtype=np.int64)

    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1

    if values.min() < lower or values.max() > upper:
        raise OverflowError(f"{name} does not fit in {bits} signed bits.")


def read_signed_mem(filename, bits, shape):
    path = EXPORT_FOLDER / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing exported parameter file: {path}\n"
            "Run export_snn_fpga.py first."
        )

    lines = path.read_text(encoding="ascii").splitlines()
    values = []

    for line in lines:
        value = int(line, 16)

        if not 0 <= value < (1 << bits):
            raise ValueError(f"Invalid value in {filename}: {line}")

        if value >= (1 << (bits - 1)):
            value -= 1 << bits

        values.append(value)

    array = np.asarray(values, dtype=np.int64)

    if array.size != int(np.prod(shape)):
        raise ValueError(f"Wrong number of values in {filename}")

    return array.reshape(shape)


def write_mem(filename, values, bits, signed):
    values = np.asarray(values, dtype=np.int64).reshape(-1)

    if signed:
        check_signed(values, bits, filename)
    elif values.min() < 0 or values.max() >= (1 << bits):
        raise OverflowError(f"{filename} does not fit in {bits} unsigned bits.")

    path = EXPORT_FOLDER / filename
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1

    with path.open("w", encoding="ascii") as file:
        for value in values:
            file.write(f"{int(value) & mask:0{digits}X}\n")

    # Verify the written values.
    decoded = []

    for line in path.read_text(encoding="ascii").splitlines():
        value = int(line, 16)

        if signed and value >= (1 << (bits - 1)):
            value -= 1 << bits

        decoded.append(value)

    if not np.array_equal(values, np.asarray(decoded, dtype=np.int64)):
        raise RuntimeError(f"Readback failed for {filename}")

    print(f"Saved {filename}: {values.size} values")


def find_data_folder():
    candidates = [
        FOLDER / "data",
        FOLDER.parent / "data",
        Path.cwd() / "data",
    ]

    for path in candidates:
        if (path / "MNIST" / "raw").is_dir():
            return path

    return FOLDER.parent / "data"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("EXPORT MNIST IMAGE AND HIDDEN-CURRENT REFERENCES")
    print("=" * 72)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    # Read the actual exported parameters that Verilog will use.
    weights = read_signed_mem(
        "fc1_weights.mem", 16, (16, 784)
    )
    biases = read_signed_mem(
        "fc1_biases.mem", 16, (16,)
    )

    # Confirm that those exports match the selected checkpoint.
    if not np.array_equal(
        weights, quantize(state["fc1.weight"].numpy())
    ):
        raise RuntimeError("Exported hidden weights do not match checkpoint.")

    if not np.array_equal(
        biases, quantize(state["fc1.bias"].numpy())
    ):
        raise RuntimeError("Exported hidden biases do not match checkpoint.")

    dataset = torchvision.datasets.MNIST(
        root=str(find_data_folder()),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image, label = dataset[TEST_IMAGE_INDEX]

    # Same ToTensor preprocessing and pixel quantization
    # as the validated simulation.
    pixels = quantize(image.reshape(784).numpy())

    if pixels.min() < 0 or pixels.max() > SCALE:
        raise ValueError("Pixel values are outside the expected range.")

    # Products have 30 fractional bits.
    # Align the bias to that scale before adding it.
    accumulators = weights @ pixels + biases * SCALE

    check_signed(accumulators, 48, "Accumulators")
    check_signed(
        accumulators + ROUND_HALF, 48, "Rounded accumulators"
    )

    # Round once after the complete weighted sum.
    currents = (accumulators + ROUND_HALF) // SCALE
    check_signed(currents, 32, "Hidden currents")

    # Independently verify with sequential accumulation,
    # which matches the intended hardware calculation.
    for neuron in range(16):
        sequential = int(biases[neuron]) * SCALE

        for pixel in range(784):
            sequential += (
                int(pixels[pixel]) * int(weights[neuron, pixel])
            )
            check_signed([sequential], 48, "Sequential partial sum")

        if sequential != int(accumulators[neuron]):
            raise RuntimeError(f"Accumulator mismatch for neuron {neuron}")

    print(f"\nMNIST test index: {TEST_IMAGE_INDEX}")
    print(f"Correct digit: {label}")
    print(f"Pixel integer range: {pixels.min()} to {pixels.max()}")
    print()

    write_mem(
        "mnist_test_image.mem", pixels, 16, signed=False
    )
    write_mem(
        "hidden_test_accumulators.mem",
        accumulators,
        48,
        signed=True,
    )
    write_mem(
        "hidden_test_currents.mem",
        currents,
        32,
        signed=True,
    )

    print("\nExpected hidden-layer currents:")
    print(
        f"{'Neuron':<10}"
        f"{'Integer current':>20}"
        f"{'Real current':>18}"
    )

    for neuron, current in enumerate(currents):
        print(
            f"{neuron:<10}"
            f"{int(current):>20}"
            f"{current / SCALE:>18.8f}"
        )

    print("\nAll readback and sequential-sum checks passed.")
    print("Existing parameter exports and checkpoint unchanged.")
    print(
        "Next: verify these 16 currents using a Verilog "
        "multiply-accumulate calculator."
    )


if __name__ == "__main__":
    main()