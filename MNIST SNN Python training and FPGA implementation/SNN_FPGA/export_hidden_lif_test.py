from pathlib import Path

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
EXPORT_FOLDER = FOLDER / "fpga_exports"

NEURONS = 16
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
            raise ValueError(f"Invalid value in {filename}: {line}")

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


def write_mem(filename, values, bits, signed):
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("EXPORT HIDDEN-LAYER LIF REFERENCES")
    print("=" * 72)

    # Read the same pixels and parameters used in the MAC test.
    pixels = read_mem(
        "mnist_test_image.mem", 16, False, 784
    )
    weights = read_mem(
        "fc1_weights.mem", 16, True, NEURONS * 784
    ).reshape(NEURONS, 784)
    biases = read_mem(
        "fc1_biases.mem", 16, True, NEURONS
    )

    expected_currents = read_mem(
        "hidden_test_currents.mem", 32, True, NEURONS
    )
    expected_accumulators = read_mem(
        "hidden_test_accumulators.mem", 48, True, NEURONS
    )

    beta = int(read_mem(
        "lif_beta.mem", 16, False, 2
    )[0])
    threshold = int(read_mem(
        "lif_threshold.mem", 32, True, 2
    )[0])

    if beta != 29491 or threshold != 32768:
        raise ValueError("Constants differ from the validated model")

    if pixels.min() < 0 or pixels.max() > SCALE:
        raise ValueError("Pixels outside the expected range")

    # Recalculate currents and verify consistency with the
    # references that already passed the Verilog MAC test.
    accumulators = weights @ pixels + biases * SCALE
    check_signed(accumulators, 48, "Hidden accumulators")

    currents = (accumulators + ROUND_HALF) // SCALE
    check_signed(currents, 32, "Hidden currents")

    if not np.array_equal(accumulators, expected_accumulators):
        raise RuntimeError("Accumulator references do not match exports")

    if not np.array_equal(currents, expected_currents):
        raise RuntimeError("Current references do not match exports")

    membrane = np.zeros(NEURONS, dtype=np.int64)

    membrane_trace = []
    spike_trace = []

    for step in range(STEPS):
        # Reset is based on the previous membrane.
        reset = (membrane > threshold).astype(np.int64)

        product = membrane * beta
        check_signed(product, 48, "Beta products")
        check_signed(product + ROUND_HALF, 48, "Rounded beta products")

        leaked = (product + ROUND_HALF) // SCALE
        integrated = leaked + currents
        check_signed(integrated, 32, "Integrated membranes")

        membrane = integrated - reset * threshold
        check_signed(membrane, 32, "Membranes")

        spike = (membrane > threshold).astype(np.int64)

        membrane_trace.append(membrane.copy())
        spike_trace.append(spike.copy())

    membranes = np.stack(membrane_trace)
    spikes = np.stack(spike_trace)

    # File order:
    # step 1: neurons 0 through 15
    # step 2: neurons 0 through 15
    # ...
    # Address = zero_based_step * 16 + neuron.
    write_mem(
        "hidden_lif_membranes.mem", membranes, 32, True
    )
    write_mem(
        "hidden_lif_spikes.mem", spikes, 8, False
    )

    print("\nHidden spikes at each SNN time step:")
    print("Each row lists neurons 0 through 15.")

    for step in range(STEPS):
        print(
            f"Step {step + 1:2d}: "
            + " ".join(str(int(value)) for value in spikes[step])
        )

    print("\nFinal membrane values:")
    print(
        f"{'Neuron':<10}"
        f"{'Integer membrane':>20}"
        f"{'Real membrane':>18}"
        f"{'Total spikes':>15}"
    )

    counts = spikes.sum(axis=0)

    for neuron in range(NEURONS):
        final_membrane = int(membranes[-1, neuron])

        print(
            f"{neuron:<10}"
            f"{final_membrane:>20}"
            f"{final_membrane / SCALE:>18.8f}"
            f"{int(counts[neuron]):>15}"
        )

    print("\nExport completed: 160 membrane and 160 spike references.")
    print("All file readback and current-consistency checks passed.")
    print("Existing exports unchanged.")


if __name__ == "__main__":
    main()