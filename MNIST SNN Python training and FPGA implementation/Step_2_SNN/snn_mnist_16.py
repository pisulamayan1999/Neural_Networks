import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF


# ============================================================
# SETTINGS
# ============================================================

batch_size = 64

# Keep 1 epoch for fair comparison with the 32-neuron network
num_epochs = 1

num_inputs = 784

# CHANGED:
# 32 hidden neurons -> 16 hidden neurons
num_hidden = 16

num_outputs = 10

# Keep the same number of time steps
num_steps = 10

beta = 0.9
learning_rate = 0.001


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


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


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# SURROGATE GRADIENT
# ============================================================

spike_grad = surrogate.fast_sigmoid(
    slope=25
)


# ============================================================
# NETWORK
# ============================================================

class SpikingMNIST(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # FIRST LAYER
        #
        # 784 pixel inputs
        #        ↓
        # 16 hidden neurons
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            num_inputs,
            num_hidden
        )

        # 16 LIF spiking neurons
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )


        # ----------------------------------------------------
        # SECOND LAYER
        #
        # 16 hidden spikes
        #        ↓
        # 10 output neurons
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            num_hidden,
            num_outputs
        )

        # 10 output LIF neurons
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Convert MNIST image:
        #
        # 28 x 28
        #
        # into:
        #
        # 784 values
        # ----------------------------------------------------

        x = x.view(
            x.size(0),
            -1
        )


        # ----------------------------------------------------
        # INITIAL MEMBRANE POTENTIALS
        # ----------------------------------------------------

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


        # ----------------------------------------------------
        # RECORD OUTPUT SPIKES
        # ----------------------------------------------------

        spk2_record = []


        # ----------------------------------------------------
        # RUN NETWORK THROUGH TIME
        # ----------------------------------------------------

        for step in range(num_steps):

            # =================================================
            # HIDDEN LAYER
            # =================================================

            # 784 pixel values
            #
            # Each of the 16 hidden neurons receives
            # all 784 pixel values.
            #
            # This calculates:
            #
            # pixel1 * weight1
            # +
            # pixel2 * weight2
            # +
            # ...
            # +
            # pixel784 * weight784
            # +
            # bias
            #
            # for all 16 neurons.
            # =================================================

            cur1 = self.fc1(x)


            # -------------------------------------------------
            # Send those 16 weighted currents
            # into the 16 LIF neurons
            # -------------------------------------------------

            spk1, mem1 = self.lif1(
                cur1,
                mem1
            )


            # =================================================
            # OUTPUT LAYER
            # =================================================

            # The 16 hidden neurons now produce spikes.
            #
            # Those 16 spikes are weighted and sent
            # into the 10 output neurons.
            # =================================================

            cur2 = self.fc2(spk1)


            # -------------------------------------------------
            # 10 output LIF neurons
            # -------------------------------------------------

            spk2, mem2 = self.lif2(
                cur2,
                mem2
            )


            # -------------------------------------------------
            # Save output spikes for this time step
            # -------------------------------------------------

            spk2_record.append(spk2)


        # ----------------------------------------------------
        # Convert the list into a tensor
        #
        # Shape:
        #
        # [time steps, batch size, 10 output neurons]
        # ----------------------------------------------------

        return torch.stack(
            spk2_record
        )


# ============================================================
# CREATE NETWORK
# ============================================================

net = SpikingMNIST().to(device)


print("\nNetwork:")
print(net)


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in net.parameters()
)


print(
    "\nTotal trainable parameters:",
    total_parameters
)


# ============================================================
# LOSS FUNCTION
# ============================================================

# This compares the spike counts from the
# 10 output neurons with the correct MNIST label.

loss_function = SF.ce_count_loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=learning_rate
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")


for epoch in range(num_epochs):

    net.train()

    running_loss = 0.0


    for batch_index, (images, labels) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move data to CPU/GPU
        # ----------------------------------------------------

        images = images.to(device)
        labels = labels.to(device)


        # ----------------------------------------------------
        # Clear gradients from previous batch
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Run images through the SNN
        #
        # The network runs for 10 time steps
        # ----------------------------------------------------

        spikes = net(images)


        # ----------------------------------------------------
        # Calculate training loss
        # ----------------------------------------------------

        loss = loss_function(
            spikes,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights and biases
        # ----------------------------------------------------

        optimizer.step()


        running_loss += loss.item()


        # ----------------------------------------------------
        # Print progress every 100 batches
        # ----------------------------------------------------

        if (batch_index + 1) % 100 == 0:

            average_loss = (
                running_loss / 100
            )

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss = {average_loss:.4f}"
            )

            running_loss = 0.0


# ============================================================
# TEST NETWORK
# ============================================================

print("\nTraining finished.")
print("\nTesting SNN...\n")


net.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        # ----------------------------------------------------
        # Run test images through network
        # ----------------------------------------------------

        spikes = net(images)


        # ----------------------------------------------------
        # Add spikes from all 10 time steps
        #
        # Original shape:
        #
        # [time steps, batch size, 10]
        #
        # After summing time:
        #
        # [batch size, 10]
        # ----------------------------------------------------

        spike_counts = spikes.sum(
            dim=0
        )


        # ----------------------------------------------------
        # Select output neuron with most spikes
        # ----------------------------------------------------

        predicted = spike_counts.argmax(
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = (
    100.0 * correct / total
)


print("==============================")
print(f"Test Accuracy = {accuracy:.2f}%")
print("==============================")


# ============================================================
# TEST ONE MNIST IMAGE
# ============================================================

images, labels = next(
    iter(test_loader)
)


# Take first image from batch
image = images[0:1].to(device)

correct_digit = labels[0].item()


with torch.no_grad():

    # Run one image through SNN
    spikes = net(image)


    # Count spikes from each output neuron
    spike_counts = spikes.sum(
        dim=0
    )[0]


    # Neuron with most spikes wins
    prediction = spike_counts.argmax().item()


# ============================================================
# SHOW RESULT
# ============================================================

print("\nExample:")
print("--------------------")

print(
    "Correct digit  :",
    correct_digit
)

print(
    "Predicted digit:",
    prediction
)


print("\nOutput neuron spike counts:")


for digit in range(10):

    print(
        f"Digit {digit}: "
        f"{int(spike_counts[digit].item())} spikes"
    )


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

torch.save(
    net.state_dict(),
    "snn_mnist_16.pth"
)


print(
    "\nSaved model as "
    "snn_mnist_16.pth"
)