from pathlib import Path

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
EXPORT_FOLDER = FOLDER / "fpga_exports"

STEPS = 10
HIDDEN_NEURONS = 16
OUTPUT_NEURONS = 10

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

    path = EXPORT_FOLDER / filename
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4

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
    print("EXPORT OUTPUT-LAYER REFERENCES")
    print("=" * 72)

    hidden_spikes = read_mem(
        "hidden_lif_spikes.mem",
        8,
        False,
        STEPS * HIDDEN_NEURONS,
    ).reshape(STEPS, HIDDEN_NEURONS)

    if not np.all((hidden_spikes == 0) | (hidden_spikes == 1)):
        raise ValueError("Hidden spikes must be binary")

    weights = read_mem(
        "fc2_weights.mem",
        16,
        True,
        OUTPUT_NEURONS * HIDDEN_NEURONS,
    ).reshape(OUTPUT_NEURONS, HIDDEN_NEURONS)

    biases = read_mem(
        "fc2_biases.mem", 16, True, OUTPUT_NEURONS
    )

    # Second entry holds the output-layer constant.
    beta = int(read_mem(
        "lif_beta.mem", 16, False, 2
    )[1])

    threshold = int(read_mem(
        "lif_threshold.mem", 32, True, 2
    )[1])

    if beta != 29491 or threshold != 32768:
        raise ValueError("Constants differ from the validated model")

    # Spikes are plain binary values, not Q15 pixels.
    # Each contribution is therefore weight or zero.
    # Currents already have 15 fractional bits: no shift needed.
    currents = hidden_spikes @ weights.T + biases
    check_signed(currents, 32, "Output currents")

    # Verify every sequential partial sum as well.
    for step in range(STEPS):
        for neuron in range(OUTPUT_NEURONS):
            sequential = int(biases[neuron])

            for hidden in range(HIDDEN_NEURONS):
                sequential += (
                    int(hidden_spikes[step, hidden])
                    * int(weights[neuron, hidden])
                )
                check_signed(
                    [sequential], 32, "Output partial sum"
                )

            if sequential != int(currents[step, neuron]):
                raise RuntimeError("Sequential current mismatch")

    membrane = np.zeros(OUTPUT_NEURONS, dtype=np.int64)

    membrane_trace = []
    spike_trace = []

    for step in range(STEPS):
        # Delayed subtraction reset from previous membrane.
        reset = (membrane > threshold).astype(np.int64)

        product = membrane * beta
        check_signed(product, 48, "Beta products")
        check_signed(
            product + ROUND_HALF, 48, "Rounded beta products"
        )

        leaked = (product + ROUND_HALF) // SCALE
        integrated = leaked + currents[step]
        check_signed(integrated, 32, "Integrated membranes")

        membrane = integrated - reset * threshold
        check_signed(membrane, 32, "Output membranes")

        spike = (membrane > threshold).astype(np.int64)

        membrane_trace.append(membrane.copy())
        spike_trace.append(spike.copy())

    membranes = np.stack(membrane_trace)
    spikes = np.stack(spike_trace)

    # Address = zero_based_step * 10 + output_neuron.
    write_mem(
        "output_test_currents.mem", currents, 32, True
    )
    write_mem(
        "output_lif_membranes.mem", membranes, 32, True
    )
    write_mem(
        "output_lif_spikes.mem", spikes, 8, False
    )

    counts = spikes.sum(axis=0)
    prediction = int(counts.argmax())

    write_mem(
        "output_test_counts.mem", counts, 8, False
    )
    write_mem(
        "output_test_prediction.mem", [prediction], 8, False
    )

    print("\nOutput spikes at each time step:")
    print("Each row lists digits 0 through 9.")

    for step in range(STEPS):
        print(
            f"Step {step + 1:2d}: "
            + " ".join(str(int(value)) for value in spikes[step])
        )

    print("\nFinal output spike counts:")

    for digit, count in enumerate(counts):
        print(f"  Digit {digit}: {int(count)} spikes")

    print(f"\nPredicted digit: {prediction}")

    # Known reference for the current digit-7 test image.
    known_counts = np.array(
        [0, 0, 0, 4, 0, 0, 0, 10, 0, 1],
        dtype=np.int64,
    )

    if not np.array_equal(counts, known_counts):
        raise RuntimeError(
            "Counts differ from the validated digit-7 reference. "
            "Check that all test exports belong to the same image."
        )

    print("\nCounts match the validated digit-7 reference.")
    print("All readback, partial-sum, and overflow checks passed.")
    print("Existing exports unchanged.")


if __name__ == "__main__":
    main()