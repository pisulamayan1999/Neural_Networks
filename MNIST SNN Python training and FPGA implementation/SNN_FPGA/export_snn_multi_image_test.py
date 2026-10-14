from pathlib import Path

import numpy as np
import torchvision
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
EXPORT_FOLDER = FOLDER / "fpga_exports"

NUM_IMAGES = 20
STEPS = 10

SCALE = 32768
ROUND_HALF = 16384


# ============================================================
# FILE HELPERS
# ============================================================

def read_mem(filename, bits, signed, count):
    path = EXPORT_FOLDER / filename

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    values = []

    for line in path.read_text(encoding="ascii").splitlines():
        value = int(line, 16)

        if not 0 <= value < (1 << bits):
            raise ValueError(f"Invalid value in {filename}")

        if signed and value >= (1 << (bits - 1)):
            value -= 1 << bits

        values.append(value)

    if len(values) != count:
        raise ValueError(
            f"{filename}: expected {count} values, "
            f"found {len(values)}"
        )

    return np.asarray(values, dtype=np.int64)


def check_signed(values, bits, name):
    values = np.asarray(values, dtype=np.int64)

    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1

    if values.min() < lower or values.max() > upper:
        raise OverflowError(f"{name} exceeds {bits} signed bits")


def write_mem(filename, values, bits, signed=False):
    values = np.asarray(values, dtype=np.int64).reshape(-1)

    if signed:
        check_signed(values, bits, filename)
    elif values.min() < 0 or values.max() >= (1 << bits):
        raise OverflowError(f"{filename} exceeds {bits} unsigned bits")

    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    path = EXPORT_FOLDER / filename

    with path.open("w", encoding="ascii") as file:
        for value in values:
            file.write(f"{int(value) & mask:0{digits}X}\n")

    recovered = read_mem(
        filename, bits, signed, values.size
    )

    if not np.array_equal(values, recovered):
        raise RuntimeError(f"Readback failed: {filename}")

    print(f"Saved {filename}: {values.size} values")


def find_data_folder():
    for path in (
        FOLDER / "data",
        FOLDER.parent / "data",
        Path.cwd() / "data",
    ):
        if (path / "MNIST" / "raw").is_dir():
            return path

    return FOLDER.parent / "data"


# ============================================================
# INTEGER LIF UPDATE
# ============================================================

def lif_step(membrane, current, beta, threshold):
    reset = (membrane > threshold).astype(np.int64)

    product = membrane * beta
    check_signed(product, 48, "Beta product")
    check_signed(product + ROUND_HALF, 48, "Rounded beta product")

    leaked = (product + ROUND_HALF) // SCALE

    integrated = leaked + current
    check_signed(integrated, 32, "Integrated membrane")

    updated = integrated - reset * threshold
    check_signed(updated, 32, "Membrane")

    spikes = (updated > threshold).astype(np.int64)

    return spikes, updated


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("EXPORT MULTIPLE MNIST IMAGES FOR VERILOG TESTING")
    print("=" * 72)

    w1 = read_mem(
        "fc1_weights.mem", 16, True, 16 * 784
    ).reshape(16, 784)
    b1 = read_mem("fc1_biases.mem", 16, True, 16)

    w2 = read_mem(
        "fc2_weights.mem", 16, True, 10 * 16
    ).reshape(10, 16)
    b2 = read_mem("fc2_biases.mem", 16, True, 10)

    betas = read_mem("lif_beta.mem", 16, False, 2)
    thresholds = read_mem("lif_threshold.mem", 32, True, 2)

    if not np.array_equal(betas, [29491, 29491]):
        raise ValueError("Unexpected beta constants")

    if not np.array_equal(thresholds, [32768, 32768]):
        raise ValueError("Unexpected thresholds")

    dataset = torchvision.datasets.MNIST(
        root=str(find_data_folder()),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image_pixels = []
    labels = []
    predictions = []
    final_counts = []

    hidden_membranes = []
    hidden_spikes = []
    output_currents = []
    output_membranes = []
    output_spikes = []

    correct = 0

    print("\nUsing MNIST test images 0 through 19:")
    print(f"{'Index':<8}{'Label':<8}{'Prediction':<12}{'Correct'}")

    for image_index in range(NUM_IMAGES):
        image, label = dataset[image_index]

        pixels = np.rint(
            image.reshape(784).numpy().astype(np.float64) * SCALE
        ).astype(np.int64)

        if pixels.min() < 0 or pixels.max() > SCALE:
            raise ValueError("Unexpected pixel range")

        accumulator1 = w1 @ pixels + b1 * SCALE
        check_signed(accumulator1, 48, "Hidden accumulator")
        check_signed(
            accumulator1 + ROUND_HALF, 48, "Rounded accumulator"
        )

        current1 = (accumulator1 + ROUND_HALF) // SCALE
        check_signed(current1, 32, "Hidden current")

        # Independent zero state for each new image.
        mem1 = np.zeros(16, dtype=np.int64)
        mem2 = np.zeros(10, dtype=np.int64)
        counts = np.zeros(10, dtype=np.int64)

        for step in range(STEPS):
            spike1, mem1 = lif_step(
                mem1, current1, int(betas[0]), int(thresholds[0])
            )

            current2 = w2 @ spike1 + b2
            check_signed(current2, 32, "Output current")

            spike2, mem2 = lif_step(
                mem2, current2, int(betas[1]), int(thresholds[1])
            )

            counts += spike2

            hidden_membranes.append(mem1.copy())
            hidden_spikes.append(spike1.copy())
            output_currents.append(current2.copy())
            output_membranes.append(mem2.copy())
            output_spikes.append(spike2.copy())

        prediction = int(counts.argmax())
        correct += int(prediction == label)

        image_pixels.append(pixels)
        labels.append(label)
        predictions.append(prediction)
        final_counts.append(counts.copy())

        print(
            f"{image_index:<8}"
            f"{label:<8}"
            f"{prediction:<12}"
            f"{prediction == label}"
        )

    # Ordering:
    # Pixels: image, then pixel.
    # Traces: image, then time step, then neuron.
    # Counts: image, then output digit.
    write_mem("multi_images.mem", np.stack(image_pixels), 16)
    write_mem("multi_labels.mem", labels, 8)
    write_mem("multi_predictions.mem", predictions, 8)
    write_mem("multi_counts.mem", np.stack(final_counts), 8)

    write_mem(
        "multi_hidden_membranes.mem",
        np.stack(hidden_membranes),
        32,
        signed=True,
    )
    write_mem(
        "multi_hidden_spikes.mem",
        np.stack(hidden_spikes),
        8,
    )
    write_mem(
        "multi_output_currents.mem",
        np.stack(output_currents),
        32,
        signed=True,
    )
    write_mem(
        "multi_output_membranes.mem",
        np.stack(output_membranes),
        32,
        signed=True,
    )
    write_mem(
        "multi_output_spikes.mem",
        np.stack(output_spikes),
        8,
    )

    print(f"\nPython classification: {correct}/{NUM_IMAGES} correct.")
    print(
        "This small sample is a hardware verification set, "
        "not a replacement for the full MNIST accuracy test."
    )
    print("All export readback checks passed.")
    print("Existing single-image exports unchanged.")


if __name__ == "__main__":
    main()