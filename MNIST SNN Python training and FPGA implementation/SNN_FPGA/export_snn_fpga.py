from pathlib import Path
import json
import hashlib

import numpy as np
import torch


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
CHECKPOINT = FOLDER / "snn_mnist_16_5epochs.pth"
OUTPUT = FOLDER / "fpga_exports"

FRACTIONAL_BITS = 15
SCALE = 1 << FRACTIONAL_BITS
HALF = SCALE // 2


# ============================================================
# FIXED-POINT HELPERS
# ============================================================

def quantize(values):
    # Same parameter rounding as simulate_snn_fixed_point.py:
    # nearest integer, halfway cases to even.
    return np.rint(
        np.asarray(values, dtype=np.float64) * SCALE
    ).astype(np.int64)


def check_range(values, bits, signed=True):
    values = np.asarray(values, dtype=np.int64)

    if signed:
        lower = -(1 << (bits - 1))
        upper = (1 << (bits - 1)) - 1
    else:
        lower = 0
        upper = (1 << bits) - 1

    if values.min() < lower or values.max() > upper:
        raise OverflowError(
            f"Values do not fit in {bits} bits: "
            f"{values.min()} to {values.max()}"
        )


def write_mem(filename, values, bits, signed=True):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    check_range(values, bits, signed)

    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    path = OUTPUT / filename

    # Negative numbers are written as two's-complement hex.
    with path.open("w", encoding="ascii") as file:
        for value in values:
            file.write(f"{int(value) & mask:0{digits}X}\n")

    # Verify the complete written file by decoding it.
    lines = path.read_text(encoding="ascii").splitlines()

    decoded = []
    for line in lines:
        value = int(line, 16)

        if signed and value >= (1 << (bits - 1)):
            value -= 1 << bits

        decoded.append(value)

    if not np.array_equal(
        np.asarray(decoded, dtype=np.int64), values
    ):
        raise RuntimeError(f"Readback verification failed: {filename}")

    print(f"  {filename:<30} {len(values):>6} values")
    return {
        "filename": filename,
        "values": len(values),
        "bits": bits,
        "signed": signed,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# ============================================================
# ONE LIF NEURON: INTEGER REFERENCE
# ============================================================

def generate_lif_trace(currents, beta, threshold):
    membrane = 0
    membranes = []
    spikes = []

    for current in currents:
        # Delayed subtraction reset:
        # reset depends on the previous membrane.
        reset = int(membrane > threshold)

        product = membrane * beta
        check_range([product, product + HALF], 48)

        # Same rounding as the validated integer simulation.
        leaked = (product + HALF) // SCALE

        integrated = leaked + int(current)
        check_range([integrated], 32)

        membrane = integrated - reset * threshold
        check_range([membrane], 32)

        spike = int(membrane > threshold)

        membranes.append(membrane)
        spikes.append(spike)

    return (
        np.asarray(membranes, dtype=np.int64),
        np.asarray(spikes, dtype=np.int64),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("EXPORT VALIDATED SNN PARAMETERS FOR FPGA")
    print("=" * 72)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Put snn_mnist_16_5epochs.pth beside this script."
        )

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    expected_shapes = {
        "fc1.weight": (16, 784),
        "fc1.bias": (16,),
        "fc2.weight": (10, 16),
        "fc2.bias": (10,),
    }

    quantized = {}

    for name, expected_shape in expected_shapes.items():
        tensor = state[name]

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name}: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains non-finite values.")

        quantized[name] = quantize(tensor.numpy())
        check_range(quantized[name], 16)

    for layer in ("lif1", "lif2"):
        if int(state[f"{layer}.reset_mechanism_val"].item()) != 0:
            raise ValueError(f"{layer}: expected subtraction reset.")

        if state[f"{layer}.graded_spikes_factor"].item() != 1.0:
            raise ValueError(f"{layer}: expected binary spikes.")

    beta1 = int(quantize(state["lif1.beta"].item()))
    beta2 = int(quantize(state["lif2.beta"].item()))
    threshold1 = int(quantize(state["lif1.threshold"].item()))
    threshold2 = int(quantize(state["lif2.threshold"].item()))

    if (beta1, beta2, threshold1, threshold2) != (
        29491, 29491, 32768, 32768
    ):
        raise ValueError(
            "Constants differ from the validated simulation."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput folder: {OUTPUT}")
    print("\nWriting and verifying files:")

    files = []

    # NumPy's default flattening is row-major:
    # all inputs of neuron 0, then all inputs of neuron 1, etc.
    files.append(write_mem(
        "fc1_weights.mem", quantized["fc1.weight"], 16
    ))
    files.append(write_mem(
        "fc1_biases.mem", quantized["fc1.bias"], 16
    ))
    files.append(write_mem(
        "fc2_weights.mem", quantized["fc2.weight"], 16
    ))
    files.append(write_mem(
        "fc2_biases.mem", quantized["fc2.bias"], 16
    ))

    # One entry per layer: hidden first, output second.
    files.append(write_mem(
        "lif_beta.mem", [beta1, beta2], 16, signed=False
    ))
    files.append(write_mem(
        "lif_threshold.mem", [threshold1, threshold2], 32
    ))

    # A single continuous sequence starting from membrane=0.
    # It exercises:
    # - exact threshold equality
    # - delayed reset
    # - repeated spikes
    # - negative currents and negative membranes
    # - leak/rounding
    # - currents comparable to those measured in MNIST
    test_currents_real = np.array([
        1.0,
        0.0,
        0.2,
        0.0,
        1.5,
        1.5,
        0.0,
        -2.0,
        -2.0,
        0.0,
        0.0,
        0.125,
        -0.125,
        5.0,
        0.0,
        0.0,
        -13.0,
        0.0,
        0.0,
        0.0,
        5.0,
        5.0,
        5.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=np.float64)

    test_currents = quantize(test_currents_real)

    membranes, spikes = generate_lif_trace(
        test_currents, beta1, threshold1
    )

    # First input is exactly 1.0 from zero membrane:
    # strict ">" comparison must produce no spike.
    if membranes[0] != threshold1 or spikes[0] != 0:
        raise RuntimeError("Threshold equality test failed.")

    files.append(write_mem(
        "lif_test_currents.mem", test_currents, 32
    ))
    files.append(write_mem(
        "lif_test_membranes.mem", membranes, 32
    ))

    # Spike values are stored as 8-bit hex for convenient files.
    # The future Verilog neuron still uses a 1-bit spike output.
    files.append(write_mem(
        "lif_test_spikes.mem", spikes, 8, signed=False
    ))

    trace_path = OUTPUT / "lif_test_trace.csv"

    with trace_path.open("w", encoding="ascii") as file:
        file.write(
            "step,current_integer,current_real,"
            "membrane_integer,membrane_real,spike\n"
        )

        for index, (current, membrane, spike) in enumerate(
            zip(test_currents, membranes, spikes), start=1
        ):
            file.write(
                f"{index},{int(current)},{current / SCALE:.8f},"
                f"{int(membrane)},{membrane / SCALE:.8f},"
                f"{int(spike)}\n"
            )

    metadata = {
        "checkpoint": CHECKPOINT.name,
        "checkpoint_sha256": hashlib.sha256(
            CHECKPOINT.read_bytes()
        ).hexdigest(),
        "network": [784, 16, 10],
        "inference_steps": 10,
        "fractional_bits": FRACTIONAL_BITS,
        "scale": SCALE,
        "weight_bias_bits": 16,
        "current_membrane_bits": 32,
        "hidden_accumulator_bits": 48,
        "hidden_accumulator_fractional_bits": 30,
        "beta_integer": [beta1, beta2],
        "threshold_integer": [threshold1, threshold2],
        "reset_mechanism": "subtract",
        "reset_delay": True,
        "spike_comparison": "membrane > threshold",
        "parameter_rounding": "nearest; ties to even",
        "product_rescaling": "(value + 16384) arithmetic shift right 15",
        "product_rounding": "nearest; ties toward positive infinity",
        "prediction_tie_break": "lowest digit index",
        "fc1_weight_address": "hidden_neuron * 784 + input_pixel",
        "fc2_weight_address": "output_neuron * 16 + hidden_neuron",
        "lif_test_steps": len(test_currents),
        "lif_test_initial_membrane": 0,
        "files": files,
    }

    (OUTPUT / "export_info.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nLIF reference trace:")
    print(
        f"{'Step':<6}"
        f"{'Current':>12}"
        f"{'Membrane':>14}"
        f"{'Spike':>8}"
    )

    for index, (current, membrane, spike) in enumerate(
        zip(test_currents, membranes, spikes), start=1
    ):
        print(
            f"{index:<6}"
            f"{current / SCALE:>12.6f}"
            f"{membrane / SCALE:>14.6f}"
            f"{int(spike):>8}"
        )

    print("\nExport completed.")
    print("All .mem files passed integer readback verification.")
    print("Original checkpoint unchanged.")
    print(
        "Next: implement one Verilog LIF neuron and compare "
        "every step against these reference vectors."
    )


if __name__ == "__main__":
    main()