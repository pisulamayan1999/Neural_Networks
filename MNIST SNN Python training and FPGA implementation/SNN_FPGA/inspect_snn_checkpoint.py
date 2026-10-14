from pathlib import Path
from collections.abc import Mapping

import torch


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "snn_mnist_16_5epochs.pth"
)

# These are the settings you reported from training.
# They are NOT treated as verified checkpoint values.
REPORTED_BETA = 0.9
REPORTED_THRESHOLD = 1.0

EXPECTED_PARAMETER_COUNT = 12730


# ============================================================
# Helper functions
# ============================================================

def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}\n\n"
            "Put this script in the same folder as "
            "snn_mnist_16_5epochs.pth."
        )

    # A state_dict can normally be loaded with weights_only=True.
    # We deliberately do not automatically load arbitrary pickled
    # model objects.
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Your PyTorch version does not support weights_only=True. "
            "Please send the error output so we can adjust the loader "
            "for your environment."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Could not load this checkpoint as a weights-only checkpoint.\n"
            "It may contain a complete model object rather than a state_dict.\n"
            "Please send this error and the original training code."
        ) from exc


def find_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "The checkpoint is not a dictionary. "
            "Please send the original training code."
        )

    # Common formats used when saving checkpoints.
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            if any(torch.is_tensor(v) for v in candidate.values()):
                return candidate, key

    # A directly saved state_dict.
    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, "checkpoint itself"

    raise RuntimeError(
        "Could not locate the model state_dict.\n"
        f"Top-level keys: {list(checkpoint.keys())}"
    )


def print_tensor_statistics(name, tensor):
    values = tensor.detach().cpu().to(torch.float64).reshape(-1)

    print(f"\n{name}")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Stored dtype: {tensor.dtype}")
    print(f"  Number of values: {values.numel()}")

    if values.numel() == 0:
        print("  Empty tensor")
        return

    if not torch.isfinite(values).all():
        print("  WARNING: contains NaN or infinity")
        return

    minimum = values.min().item()
    maximum = values.max().item()

    print(f"  Minimum: {minimum:.10f}")
    print(f"  Maximum: {maximum:.10f}")
    print(f"  Maximum absolute value: {values.abs().max().item():.10f}")
    print(f"  Mean: {values.mean().item():.10f}")
    print(f"  Standard deviation: {values.std(unbiased=False).item():.10f}")

    if values.numel() <= 16:
        print(f"  Values: {values.tolist()}")


def inspect_q15(name, tensor):
    values = tensor.detach().cpu().to(torch.float64).reshape(-1)

    if values.numel() == 0 or not torch.isfinite(values).all():
        return

    scale = 2 ** 15
    minimum_representable = -1.0
    maximum_representable = 32767 / scale

    # Evaluate round-to-nearest conversion without writing or
    # modifying any model data.
    rounded_integers = torch.round(values * scale)

    overflow = (
        (rounded_integers < -32768)
        | (rounded_integers > 32767)
    )

    reconstructed = torch.clamp(
        rounded_integers,
        -32768,
        32767,
    ) / scale

    errors = reconstructed - values

    print(f"\n{name}")
    print(
        "  Q1.15 representable range: "
        f"{minimum_representable:.10f} to "
        f"{maximum_representable:.10f}"
    )
    print(
        "  Values outside the representable range: "
        f"{((values < minimum_representable) | (values > maximum_representable)).sum().item()}"
    )
    print(
        "  Values overflowing after rounding: "
        f"{overflow.sum().item()}"
    )
    print(
        "  Maximum conversion error "
        "(including clipping if needed): "
        f"{errors.abs().max().item():.10f}"
    )
    print(
        "  Mean absolute conversion error: "
        f"{errors.abs().mean().item():.10f}"
    )


def inspect_lif_settings(state_dict):
    print("\n" + "=" * 72)
    print("LIF SETTINGS STORED IN CHECKPOINT")
    print("=" * 72)

    for setting in ("beta", "threshold"):
        matches = [
            (name, value)
            for name, value in state_dict.items()
            if torch.is_tensor(value)
            and name.split(".")[-1].lower() == setting
        ]

        if matches:
            for name, value in matches:
                print_tensor_statistics(name, value)
        else:
            reported = (
                REPORTED_BETA
                if setting == "beta"
                else REPORTED_THRESHOLD
            )
            print(
                f"\nNo '{setting}' tensor found.\n"
                f"  Reported training value: {reported}\n"
                "  Confirm this in the original training code."
            )

    print("\nOther potentially relevant stored settings:")

    found = False

    for name, value in state_dict.items():
        lowered = name.lower()

        if any(
            word in lowered
            for word in ("reset", "graded", "learn")
        ):
            found = True
            if torch.is_tensor(value):
                print_tensor_statistics(name, value)
            else:
                print(f"  {name}: {value}")

    if not found:
        print("  No additional settings found by name.")

    print(
        "\nThe checkpoint alone may not specify input encoding, "
        "reset timing, or the forward-pass sequence."
    )


# ============================================================
# Main inspection
# ============================================================

def main():
    print("=" * 72)
    print("SNN CHECKPOINT INSPECTION")
    print("=" * 72)
    print(f"File: {CHECKPOINT_PATH}")
    print(f"PyTorch version: {torch.__version__}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    state_dict, source = find_state_dict(checkpoint)

    print(f"\nState dictionary located in: {source}")

    print("\nStored tensor names:")
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            print(
                f"  {name}: "
                f"shape={tuple(value.shape)}, dtype={value.dtype}"
            )

    parameter_tensors = [
        (name, value)
        for name, value in state_dict.items()
        if torch.is_tensor(value)
        and name.split(".")[-1] in ("weight", "bias")
    ]

    print("\n" + "=" * 72)
    print("WEIGHT AND BIAS STATISTICS")
    print("=" * 72)

    parameter_count = sum(
        tensor.numel()
        for _, tensor in parameter_tensors
    )

    for name, tensor in parameter_tensors:
        print_tensor_statistics(name, tensor)

    print(f"\nTotal weight/bias values found: {parameter_count}")
    print(f"Expected for 784 -> 16 -> 10: {EXPECTED_PARAMETER_COUNT}")

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        print(
            "WARNING: parameter count differs from the expected model. "
            "Check the tensor names and original training code."
        )

    inspect_lif_settings(state_dict)

    print("\n" + "=" * 72)
    print("PRELIMINARY Q1.15 CHECK FOR WEIGHTS AND BIASES")
    print("=" * 72)

    for name, tensor in parameter_tensors:
        inspect_q15(name, tensor)

    print("\n" + "=" * 72)
    print("WHAT STILL NEEDS TO BE MEASURED")
    print("=" * 72)
    print(
        "Using the exact trained model and MNIST preprocessing, "
        "we must record:\n"
        "  1. Input pixel range\n"
        "  2. Hidden-layer weighted-current range\n"
        "  3. Hidden-layer membrane range across all 10 time steps\n"
        "  4. Output-layer weighted-current range\n"
        "  5. Output-layer membrane range across all 10 time steps\n"
        "  6. Spike counts and the final prediction rule\n\n"
        "These require the original model definition and forward pass.\n"
        "No weights were changed and no export files were created."
    )


if __name__ == "__main__":
    main()