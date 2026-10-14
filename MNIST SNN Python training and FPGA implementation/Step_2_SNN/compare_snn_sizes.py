import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

import random
import numpy as np


# ============================================================
# GLOBAL SETTINGS
# ============================================================

batch_size = 64

# Now train properly instead of only 1 epoch
num_epochs = 5

num_inputs = 784
num_outputs = 10

num_steps = 10

beta = 0.9
learning_rate = 0.001

# Hidden layer sizes we want to compare
hidden_sizes = [
    32,
    16,
    8
]

# Fixed seed so results are reproducible
random_seed = 42


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# FIX RANDOM SEEDS
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD MNIST
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# SPIKE GRADIENT
# ============================================================

spike_grad = surrogate.fast_sigmoid(
    slope=25
)


# ============================================================
# SNN CLASS
# ============================================================

class SpikingMNIST(nn.Module):

    def __init__(self, hidden_neurons):

        super().__init__()


        # =====================================================
        # 784 INPUTS -> HIDDEN LAYER
        # =====================================================

        self.fc1 = nn.Linear(
            num_inputs,
            hidden_neurons
        )


        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )


        # =====================================================
        # HIDDEN LAYER -> 10 OUTPUT NEURONS
        # =====================================================

        self.fc2 = nn.Linear(
            hidden_neurons,
            num_outputs
        )


        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )


    def forward(self, x):


        # =====================================================
        # FLATTEN MNIST IMAGE
        #
        # 28 x 28 -> 784
        # =====================================================

        x = x.view(
            x.size(0),
            -1
        )


        # =====================================================
        # INITIAL MEMBRANE POTENTIALS
        # =====================================================

        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()


        # =====================================================
        # RECORD OUTPUT SPIKES
        # =====================================================

        output_spikes = []


        # =====================================================
        # RUN NETWORK THROUGH TIME
        # =====================================================

        for step in range(num_steps):


            # -------------------------------------------------
            # 784 PIXELS -> HIDDEN CURRENT
            # -------------------------------------------------

            current1 = self.fc1(
                x
            )


            # -------------------------------------------------
            # HIDDEN LIF NEURONS
            # -------------------------------------------------

            spike1, mem1 = self.lif1(
                current1,
                mem1
            )


            # -------------------------------------------------
            # HIDDEN SPIKES -> 10 OUTPUT CURRENTS
            # -------------------------------------------------

            current2 = self.fc2(
                spike1
            )


            # -------------------------------------------------
            # OUTPUT LIF NEURONS
            # -------------------------------------------------

            spike2, mem2 = self.lif2(
                current2,
                mem2
            )


            # -------------------------------------------------
            # STORE OUTPUT SPIKES
            # -------------------------------------------------

            output_spikes.append(
                spike2
            )


        return torch.stack(
            output_spikes
        )


# ============================================================
# FUNCTION TO TRAIN ONE NETWORK
# ============================================================

def train_network(hidden_neurons):


    print()
    print("============================================")
    print(
        f"TRAINING NETWORK: "
        f"784 -> {hidden_neurons} -> 10"
    )
    print("============================================")


    # --------------------------------------------------------
    # RESET RANDOM SEED
    #
    # This makes each experiment reproducible.
    # --------------------------------------------------------

    set_seed(
        random_seed
    )


    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_generator = torch.Generator()

    train_generator.manual_seed(
        random_seed
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator
    )


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # --------------------------------------------------------
    # CREATE NETWORK
    # --------------------------------------------------------

    net = SpikingMNIST(
        hidden_neurons
    ).to(device)


    # --------------------------------------------------------
    # COUNT PARAMETERS
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in net.parameters()
    )


    print(
        "Trainable parameters:",
        total_parameters
    )


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    loss_function = SF.ce_count_loss()


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate
    )


    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(num_epochs):


        net.train()


        running_loss = 0.0


        for batch_index, (images, labels) in enumerate(
            train_loader
        ):


            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            # ------------------------------------------------
            # CLEAR PREVIOUS GRADIENTS
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            spikes = net(
                images
            )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = loss_function(
                spikes,
                labels
            )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # UPDATE WEIGHTS
            # ------------------------------------------------

            optimizer.step()


            running_loss += loss.item()


        # ----------------------------------------------------
        # AVERAGE LOSS FOR THIS EPOCH
        # ----------------------------------------------------

        average_loss = (
            running_loss /
            len(train_loader)
        )


        print(
            f"Epoch "
            f"{epoch + 1}/{num_epochs} | "
            f"Average Loss = "
            f"{average_loss:.4f}"
        )


    # ========================================================
    # TEST NETWORK
    # ========================================================

    print(
        "Testing..."
    )


    net.eval()


    correct = 0

    total = 0


    with torch.no_grad():


        for images, labels in test_loader:


            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            # ------------------------------------------------
            # RUN SNN
            # ------------------------------------------------

            spikes = net(
                images
            )


            # ------------------------------------------------
            # COUNT SPIKES FROM EACH OUTPUT NEURON
            # ------------------------------------------------

            spike_counts = spikes.sum(
                dim=0
            )


            # ------------------------------------------------
            # NEURON WITH MOST SPIKES = PREDICTION
            # ------------------------------------------------

            predicted = spike_counts.argmax(
                dim=1
            )


            total += labels.size(
                0
            )


            correct += (
                predicted == labels
            ).sum().item()


    # ========================================================
    # CALCULATE ACCURACY
    # ========================================================

    accuracy = (
        100.0 *
        correct /
        total
    )


    print(
        f"Accuracy = "
        f"{accuracy:.2f}%"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    filename = (
        f"snn_mnist_"
        f"{hidden_neurons}_"
        f"5epochs.pth"
    )


    torch.save(
        net.state_dict(),
        filename
    )


    print(
        "Saved:",
        filename
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "hidden": hidden_neurons,
        "parameters": total_parameters,
        "accuracy": accuracy
    }


# ============================================================
# RUN ALL THREE EXPERIMENTS
# ============================================================

results = []


for hidden_neurons in hidden_sizes:


    result = train_network(
        hidden_neurons
    )


    results.append(
        result
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print()
print("========================================================")
print("FINAL SNN SIZE COMPARISON")
print("========================================================")

print(
    f"{'Hidden':<10}"
    f"{'Parameters':<15}"
    f"{'Accuracy':<15}"
    f"{'Parameter Reduction'}"
)

print(
    "--------------------------------------------------------"
)


# Parameter count of largest model
baseline_parameters = results[0][
    "parameters"
]


for result in results:


    reduction = (
        100.0 *
        (
            baseline_parameters -
            result["parameters"]
        )
        /
        baseline_parameters
    )


    print(
        f"{result['hidden']:<10}"
        f"{result['parameters']:<15}"
        f"{result['accuracy']:<14.2f}%"
        f"{reduction:.2f}%"
    )


print(
    "========================================================"
)


# ============================================================
# FIND BEST SMALLER NETWORK
# ============================================================

print()
print("Experiment completed.")
print()

print(
    "The purpose of this experiment is to determine "
    "how much the hidden layer can be reduced while "
    "maintaining acceptable MNIST classification accuracy."
)