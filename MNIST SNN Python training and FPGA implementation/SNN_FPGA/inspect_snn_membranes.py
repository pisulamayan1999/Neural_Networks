from pathlib import Path
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import surrogate


# ============================================================
# SETTINGS
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_FOLDER / "snn_mnist_16_5epochs.pth"

DATA_CANDIDATES = [
    SCRIPT_FOLDER / "data",
    SCRIPT_FOLDER.parent / "data",
    Path.cwd() / "data",
]

BATCH_SIZE = 64
NUM_INPUTS = 784
NUM_HIDDEN = 16
NUM_OUTPUTS = 10
NUM_STEPS = 10
EXPECTED_ACCURACY = 92.58
DEVICE = torch.device("cpu")


# ============================================================
# RANGE TRACKER
# ============================================================

class RangeTracker:
    def __init__(self):
        self.ranges = {}

    def update(self, name, tensor):
        values = tensor.detach()

        if values.numel() == 0:
            return

        if not torch.isfinite(values).all().item():
            raise RuntimeError(f"Non-finite value in {name}")

        minimum = values.min().item()
        maximum = values.max().item()

        if name not in self.ranges:
            self.ranges[name] = [minimum, maximum]
        else:
            self.ranges[name][0] = min(
                self.ranges[name][0], minimum
            )
            self.ranges[name][1] = max(
                self.ranges[name][1], maximum
            )

    def print_table(self):
        print("\n" + "=" * 78)
        print("MEASURED RANGES: ALL TEST IMAGES")
        print("=" * 78)
        print(
            f"{'Quantity':<42}"
            f"{'Minimum':>16}"
            f"{'Maximum':>16}"
        )

        for name, (minimum, maximum) in self.ranges.items():
            print(
                f"{name:<42}"
                f"{minimum:>16.8f}"
                f"{maximum:>16.8f}"
            )


# ============================================================
# NETWORK
# ============================================================

class SpikingMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN)
        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
        )

        self.fc2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUTS)
        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad,
        )

    def forward(self, images, tracker, step_ranges):
        x = images.reshape(images.size(0), NUM_INPUTS)

        # Reset internal state, then create correctly shaped
        # zero membranes for this batch, including the final
        # batch of only 16 images.
        self.lif1.init_leaky()
        self.lif2.init_leaky()

        mem1 = x.new_zeros((x.size(0), NUM_HIDDEN))
        mem2 = x.new_zeros((x.size(0), NUM_OUTPUTS))

        tracker.update("Input pixels", x)
        tracker.update("Hidden returned membrane", mem1)
        tracker.update("Output returned membrane", mem2)

        output_spikes = []

        for step in range(NUM_STEPS):
            current1 = self.fc1(x)

            leaked1 = self.lif1.beta.clamp(0, 1) * mem1
            tracker.update("Hidden weighted current", current1)
            tracker.update(
                "Hidden beta * previous membrane", leaked1
            )
            tracker.update(
                "Hidden leak + current (no reset)",
                leaked1 + current1,
            )

            # Actual neuron behavior remains snnTorch's.
            spike1, mem1 = self.lif1(current1, mem1)

            current2 = self.fc2(spike1)

            leaked2 = self.lif2.beta.clamp(0, 1) * mem2
            tracker.update("Output weighted current", current2)
            tracker.update(
                "Output beta * previous membrane", leaked2
            )
            tracker.update(
                "Output leak + current (no reset)",
                leaked2 + current2,
            )

            spike2, mem2 = self.lif2(current2, mem2)

            tracker.update("Hidden returned membrane", mem1)
            tracker.update("Output returned membrane", mem2)
            tracker.update("Hidden spikes", spike1)
            tracker.update("Output spikes", spike2)

            step_ranges[step].update("Hidden membrane", mem1)
            step_ranges[step].update("Output membrane", mem2)

            output_spikes.append(spike2)

        return torch.stack(output_spikes)


# ============================================================
# HELPERS
# ============================================================

def find_data_folder():
    for folder in DATA_CANDIDATES:
        if (folder / "MNIST" / "raw").is_dir():
            return folder

    return SCRIPT_FOLDER.parent / "data"


def print_lif_settings(name, neuron):
    print(f"\n{name}:")
    print(f"  Beta: {neuron.beta.item():.10f}")
    print(f"  Threshold: {neuron.threshold.item():.10f}")
    print(
        "  Graded spike factor: "
        f"{neuron.graded_spikes_factor.item():.10f}"
    )
    print(f"  Reset mechanism: {neuron.reset_mechanism}")
    print(f"  Reset delay: {neuron.reset_delay}")


def minimum_signed_width(minimum, maximum, fractional_bits=15):
    scale = 2 ** fractional_bits
    required_minimum = math.floor(minimum * scale)
    required_maximum = math.ceil(maximum * scale)

    width = fractional_bits + 1

    while (
        required_minimum < -(2 ** (width - 1))
        or required_maximum > (2 ** (width - 1)) - 1
    ):
        width += 1

    return width


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("SNN CURRENT AND MEMBRANE INSPECTION")
    print("=" * 78)
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"snnTorch: {getattr(snn, '__version__', 'unknown')}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Put snn_mnist_16_5epochs.pth beside this script."
        )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    net = SpikingMNIST().to(DEVICE)
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    print("\nCheckpoint loaded successfully.")
    print_lif_settings("Hidden LIF layer", net.lif1)
    print_lif_settings("Output LIF layer", net.lif2)

    data_folder = find_data_folder()
    print(f"\nMNIST data folder: {data_folder}")

    test_dataset = torchvision.datasets.MNIST(
        root=str(data_folder),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    tracker = RangeTracker()
    step_ranges = [RangeTracker() for _ in range(NUM_STEPS)]

    correct = 0
    total = 0
    tied_predictions = 0
    example = None

    print("\nMeasuring test images...")

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            spikes = net(images, tracker, step_ranges)
            counts = spikes.sum(dim=0)
            predicted = counts.argmax(dim=1)

            tracker.update("Output spike counts", counts)

            maximum_counts = counts.max(
                dim=1, keepdim=True
            ).values

            ties = (
                (counts == maximum_counts).sum(dim=1) > 1
            )

            tied_predictions += ties.sum().item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if example is None:
                example = {
                    "label": labels[0].item(),
                    "prediction": predicted[0].item(),
                    "counts": counts[0].tolist(),
                }

            if (
                (batch_index + 1) % 25 == 0
                or batch_index + 1 == len(test_loader)
            ):
                print(
                    f"  Processed {total}/{len(test_dataset)} images"
                )

    accuracy = 100.0 * correct / total

    print("\n" + "=" * 78)
    print("ACCURACY CHECK")
    print("=" * 78)
    print(f"Correct: {correct}/{total}")
    print(f"Measured accuracy: {accuracy:.2f}%")
    print(f"Previously reported accuracy: {EXPECTED_ACCURACY:.2f}%")
    print(f"Images with tied highest spike counts: {tied_predictions}")

    if round(accuracy, 2) == EXPECTED_ACCURACY:
        print("Result: accuracy matches.")
    else:
        print(
            "Result: accuracy differs. Investigate before "
            "finalizing the fixed-point implementation."
        )

    tracker.print_table()

    print("\n" + "=" * 78)
    print("MEMBRANE RANGES BY TIME STEP")
    print("=" * 78)
    print(
        f"{'Step':<7}"
        f"{'Hidden min':>16}"
        f"{'Hidden max':>16}"
        f"{'Output min':>16}"
        f"{'Output max':>16}"
    )

    for step, ranges in enumerate(step_ranges):
        hidden_min, hidden_max = ranges.ranges["Hidden membrane"]
        output_min, output_max = ranges.ranges["Output membrane"]

        print(
            f"{step + 1:<7}"
            f"{hidden_min:>16.6f}"
            f"{hidden_max:>16.6f}"
            f"{output_min:>16.6f}"
            f"{output_max:>16.6f}"
        )

    print("\n" + "=" * 78)
    print("PRELIMINARY WIDTHS WITH 15 FRACTIONAL BITS")
    print("=" * 78)

    for name in (
        "Hidden weighted current",
        "Output weighted current",
        "Hidden leak + current (no reset)",
        "Output leak + current (no reset)",
        "Hidden returned membrane",
        "Output returned membrane",
    ):
        minimum, maximum = tracker.ranges[name]

        # Include initial zero and threshold=1.0.
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 1.0)

        width = minimum_signed_width(minimum, maximum)

        print(
            f"{name}: {width} signed bits minimum "
            "for the observed range"
        )

    print(
        "\nThese widths cover measured ranges only.\n"
        "Final hardware widths also need to cover partial sums,\n"
        "worst-case input bounds, and quantization effects."
    )

    print("\nFirst test image:")
    print(f"  Correct digit: {example['label']}")
    print(f"  Predicted digit: {example['prediction']}")
    print(
        "  Output spike counts: "
        f"{[int(value) for value in example['counts']]}"
    )

    print("\nInspection complete. Checkpoint unchanged.")


if __name__ == "__main__":
    main()