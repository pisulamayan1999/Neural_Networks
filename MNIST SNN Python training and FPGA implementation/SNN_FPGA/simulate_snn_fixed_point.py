from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import snntorch as snn
from snntorch import surrogate


# ============================================================
# SETTINGS
# ============================================================

FOLDER = Path(__file__).resolve().parent
CHECKPOINT = FOLDER / "snn_mnist_16_5epochs.pth"

BATCH_SIZE = 64
STEPS = 10

FRACTIONAL_BITS = 15
SCALE = 1 << FRACTIONAL_BITS
HALF = SCALE // 2

EXPECTED_ACCURACY = 92.58

# All integer calculations use int64 containers.
# We explicitly check the narrower proposed hardware widths.
# No saturation or wraparound is silently applied.


# ============================================================
# OVERFLOW AND RANGE CHECKING
# ============================================================

class Monitor:
    def __init__(self):
        self.ranges = {}

    def check(self, name, values, bits, signed=True):
        values = np.asarray(values)

        minimum = int(values.min())
        maximum = int(values.max())

        if signed:
            lower = -(1 << (bits - 1))
            upper = (1 << (bits - 1)) - 1
        else:
            lower = 0
            upper = (1 << bits) - 1

        if minimum < lower or maximum > upper:
            raise OverflowError(
                f"{name}: range [{minimum}, {maximum}] "
                f"exceeds {bits}-bit "
                f"{'signed' if signed else 'unsigned'} storage."
            )

        if name not in self.ranges:
            self.ranges[name] = [minimum, maximum]
        else:
            self.ranges[name][0] = min(
                minimum, self.ranges[name][0]
            )
            self.ranges[name][1] = max(
                maximum, self.ranges[name][1]
            )

    def print_ranges(self):
        print("\nFixed-point ranges, converted back to real values:")

        for name in (
            "Hidden current",
            "Output current",
            "Hidden membrane",
            "Output membrane",
        ):
            minimum, maximum = self.ranges[name]
            print(
                f"  {name:<20}: "
                f"{minimum / SCALE:>12.6f} to "
                f"{maximum / SCALE:>12.6f}"
            )


# ============================================================
# ROUNDING RULES
# ============================================================

def quantize(values):
    """
    Convert floating-point constants or pixels to integers.
    Round to nearest, with exact halfway cases going to even.
    """
    return np.rint(
        np.asarray(values, dtype=np.float64) * SCALE
    ).astype(np.int64)


def rescale_product(values):
    """
    Convert 30 fractional bits to 15 fractional bits.

    Round to nearest, with halfway cases toward positive infinity.
    Integer floor division handles negative values consistently.

    Equivalent hardware operation:
        (signed_value + 16384) arithmetic-shift-right 15

    The addition and shift must use a sufficiently wide signed value.
    """
    return (values + HALF) // SCALE


# ============================================================
# ORIGINAL FLOATING-POINT NETWORK
# ============================================================

class FloatSNN(nn.Module):
    def __init__(self):
        super().__init__()

        grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(784, 16)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=grad)

        self.fc2 = nn.Linear(16, 10)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=grad)

    def forward(self, images):
        x = images.reshape(images.size(0), 784)

        self.lif1.init_leaky()
        self.lif2.init_leaky()

        mem1 = x.new_zeros((x.size(0), 16))
        mem2 = x.new_zeros((x.size(0), 10))

        hidden_spikes = []
        output_spikes = []

        for _ in range(STEPS):
            current1 = self.fc1(x)
            spike1, mem1 = self.lif1(current1, mem1)

            current2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(current2, mem2)

            hidden_spikes.append(spike1)
            output_spikes.append(spike2)

        return (
            torch.stack(hidden_spikes),
            torch.stack(output_spikes),
        )


# ============================================================
# INTEGER-ONLY NETWORK
# ============================================================

class FixedSNN:
    def __init__(self, state, monitor):
        self.monitor = monitor

        self.w1 = quantize(state["fc1.weight"].numpy())
        self.b1 = quantize(state["fc1.bias"].numpy())
        self.w2 = quantize(state["fc2.weight"].numpy())
        self.b2 = quantize(state["fc2.bias"].numpy())

        self.beta1 = int(quantize(state["lif1.beta"].item()))
        self.beta2 = int(quantize(state["lif2.beta"].item()))

        self.threshold1 = int(
            quantize(state["lif1.threshold"].item())
        )
        self.threshold2 = int(
            quantize(state["lif2.threshold"].item())
        )

        for name, values in (
            ("Hidden weights", self.w1),
            ("Hidden biases", self.b1),
            ("Output weights", self.w2),
            ("Output biases", self.b2),
        ):
            monitor.check(name, values, 16)

        monitor.check("Hidden beta", [self.beta1], 16, signed=False)
        monitor.check("Output beta", [self.beta2], 16, signed=False)
        monitor.check("Hidden threshold", [self.threshold1], 32)
        monitor.check("Output threshold", [self.threshold2], 32)

        if not (
            0 <= self.beta1 <= SCALE
            and 0 <= self.beta2 <= SCALE
        ):
            raise ValueError("Beta must be between 0 and 1.")

        # Absolute-sum bounds cover every partial sum,
        # irrespective of accumulation order.
        # Input integers are in [0, SCALE].
        hidden_bound = (
            np.abs(self.w1).sum(axis=1) * SCALE
            + np.abs(self.b1) * SCALE
        )

        # Hidden spikes are binary, so output products are
        # either zero or the corresponding weight.
        output_bound = (
            np.abs(self.w2).sum(axis=1)
            + np.abs(self.b2)
        )

        monitor.check(
            "Hidden partial-sum bound", hidden_bound, 48
        )
        monitor.check(
            "Output partial-sum bound", output_bound, 32
        )

        print("\nInteger constants:")
        print(f"  Scale: {SCALE}")
        print(f"  Hidden beta: {self.beta1}")
        print(f"  Output beta: {self.beta2}")
        print(f"  Hidden threshold: {self.threshold1}")
        print(f"  Output threshold: {self.threshold2}")

        print("\nConservative accumulator bounds passed:")
        print("  Hidden: 48 signed bits, 30 fractional bits")
        print("  Output: 32 signed bits, 15 fractional bits")

    def lif_step(self, membrane, current, beta, threshold, name):
        # Reset is determined from the PREVIOUS membrane.
        reset = (membrane > threshold).astype(np.int64)

        product = membrane * beta
        self.monitor.check(f"{name} beta product", product, 48)
        self.monitor.check(
            f"{name} rounded beta product",
            product + HALF,
            48,
        )

        leaked = rescale_product(product)

        integrated = leaked + current
        self.monitor.check(f"{name} leak + current", integrated, 32)

        updated = integrated - reset * threshold
        self.monitor.check(f"{name} membrane", updated, 32)

        # Strict comparison: equality does not emit a spike.
        spike = (updated > threshold).astype(np.int64)

        return spike, updated

    def forward(self, pixels):
        self.monitor.check("Input pixels", pixels, 16, signed=False)

        if pixels.min() < 0 or pixels.max() > SCALE:
            raise ValueError("Input pixels must be between 0 and 1.")

        batch = pixels.shape[0]

        mem1 = np.zeros((batch, 16), dtype=np.int64)
        mem2 = np.zeros((batch, 10), dtype=np.int64)

        hidden_spikes = []
        output_spikes = []

        # The same pixels are supplied at every time step,
        # so this current can be calculated once per image.
        # Bias is aligned to 30 fractional bits before addition.
        accumulator1 = (
            pixels @ self.w1.T
            + self.b1 * SCALE
        )

        self.monitor.check(
            "Hidden accumulator", accumulator1, 48
        )
        self.monitor.check(
            "Hidden rounded accumulator",
            accumulator1 + HALF,
            48,
        )

        current1 = rescale_product(accumulator1)
        self.monitor.check("Hidden current", current1, 32)

        for _ in range(STEPS):
            spike1, mem1 = self.lif_step(
                mem1,
                current1,
                self.beta1,
                self.threshold1,
                "Hidden",
            )

            # A spike is an integer 0 or 1, not a Q15 pixel.
            # Therefore no rescaling is needed in this layer.
            current2 = spike1 @ self.w2.T + self.b2
            self.monitor.check("Output current", current2, 32)

            spike2, mem2 = self.lif_step(
                mem2,
                current2,
                self.beta2,
                self.threshold2,
                "Output",
            )

            hidden_spikes.append(spike1)
            output_spikes.append(spike2)

        return (
            np.stack(hidden_spikes),
            np.stack(output_spikes),
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("FLOATING-POINT VS INTEGER FIXED-POINT SNN")
    print("=" * 72)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"PyTorch: {torch.__version__}")
    print(f"snnTorch: {getattr(snn, '__version__', 'unknown')}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Put the checkpoint beside this script."
        )

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    floating = FloatSNN()
    floating.load_state_dict(state, strict=True)
    floating.eval()

    for name, neuron in (
        ("Hidden", floating.lif1),
        ("Output", floating.lif2),
    ):
        if (
            neuron.reset_mechanism != "subtract"
            or not neuron.reset_delay
            or neuron.graded_spikes_factor.item() != 1.0
        ):
            raise ValueError(
                f"{name}: unsupported LIF settings."
            )

    # Verify threshold equality behavior in the installed version.
    probe = snn.Leaky(
        beta=0.9,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        reset_mechanism="subtract",
        reset_delay=True,
    )

    with torch.no_grad():
        spike, _ = probe(torch.tensor([1.0]), torch.zeros(1))

    if spike.item() != 0:
        raise RuntimeError(
            "Installed neuron uses unexpected threshold behavior."
        )

    monitor = Monitor()
    fixed = FixedSNN(state, monitor)

    candidates = [
        FOLDER / "data",
        FOLDER.parent / "data",
        Path.cwd() / "data",
    ]

    data_folder = next(
        (
            path for path in candidates
            if (path / "MNIST" / "raw").is_dir()
        ),
        FOLDER.parent / "data",
    )

    dataset = torchvision.datasets.MNIST(
        root=str(data_folder),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    float_correct = 0
    fixed_correct = 0
    changed_predictions = 0
    lost_correct = 0
    gained_correct = 0
    changed_count_images = 0

    hidden_differences = 0
    output_differences = 0
    hidden_total = 0
    output_total = 0

    maximum_count_difference = 0
    total = 0
    example = None

    print("\nTesting all 10,000 images...")

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            float_hidden, float_output = floating(images)

            pixels = quantize(
                images.reshape(images.size(0), 784).numpy()
            )
            fixed_hidden, fixed_output = fixed.forward(pixels)

            float_hidden = float_hidden.numpy().astype(np.int64)
            float_output = float_output.numpy().astype(np.int64)

            float_counts = float_output.sum(axis=0)
            fixed_counts = fixed_output.sum(axis=0)

            monitor.check(
                "Output spike counts", fixed_counts, 4, signed=False
            )

            # Both argmax implementations choose the lowest
            # index when multiple outputs share the maximum.
            float_prediction = float_counts.argmax(axis=1)
            fixed_prediction = fixed_counts.argmax(axis=1)

            truth = labels.numpy()
            float_ok = float_prediction == truth
            fixed_ok = fixed_prediction == truth

            float_correct += int(float_ok.sum())
            fixed_correct += int(fixed_ok.sum())
            changed_predictions += int(
                (float_prediction != fixed_prediction).sum()
            )
            lost_correct += int((float_ok & ~fixed_ok).sum())
            gained_correct += int((~float_ok & fixed_ok).sum())

            changed_count_images += int(
                np.any(float_counts != fixed_counts, axis=1).sum()
            )

            maximum_count_difference = max(
                maximum_count_difference,
                int(np.abs(float_counts - fixed_counts).max()),
            )

            hidden_differences += int(
                (float_hidden != fixed_hidden).sum()
            )
            output_differences += int(
                (float_output != fixed_output).sum()
            )

            hidden_total += float_hidden.size
            output_total += float_output.size

            if example is None:
                example = (
                    int(truth[0]),
                    int(float_prediction[0]),
                    int(fixed_prediction[0]),
                    float_counts[0].tolist(),
                    fixed_counts[0].tolist(),
                )

            total += len(truth)

            if (
                (batch_index + 1) % 25 == 0
                or batch_index + 1 == len(loader)
            ):
                print(f"  Processed {total}/{len(dataset)}")

    float_accuracy = 100.0 * float_correct / total
    fixed_accuracy = 100.0 * fixed_correct / total

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(
        f"Floating-point accuracy: {float_accuracy:.2f}% "
        f"({float_correct}/{total})"
    )
    print(
        f"Fixed-point accuracy:   {fixed_accuracy:.2f}% "
        f"({fixed_correct}/{total})"
    )
    print(
        "Accuracy change: "
        f"{fixed_accuracy - float_accuracy:+.2f} percentage points"
    )

    print(f"\nChanged predictions: {changed_predictions}")
    print(f"Correct predictions lost: {lost_correct}")
    print(f"Correct predictions gained: {gained_correct}")
    print(f"Images with changed output counts: {changed_count_images}")
    print(f"Maximum output count difference: {maximum_count_difference}")

    print(
        "\nHidden spike differences: "
        f"{hidden_differences}/{hidden_total}"
    )
    print(
        "Output spike differences: "
        f"{output_differences}/{output_total}"
    )

    print("\nOverflow: none detected in checked operations.")
    print("Conservative partial-sum bounds: passed.")

    if round(float_accuracy, 2) != EXPECTED_ACCURACY:
        print(
            "\nWARNING: floating-point accuracy does not match 92.58%.\n"
            "Resolve this before exporting or implementing Verilog."
        )
    else:
        print("\nFloating-point baseline matches 92.58%.")

    monitor.print_ranges()

    truth, fp, qp, fc, qc = example
    print("\nFirst test image:")
    print(f"  Correct digit: {truth}")
    print(f"  Floating-point prediction: {fp}")
    print(f"  Fixed-point prediction: {qp}")
    print(f"  Floating-point counts: {fc}")
    print(f"  Fixed-point counts: {qc}")

    print(
        "\nFinished. No retraining, checkpoint changes, "
        "or weight exports were performed."
    )


if __name__ == "__main__":
    main()