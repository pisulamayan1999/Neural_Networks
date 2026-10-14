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

# Start with ONLY 1 epoch for testing
num_epochs = 1

num_inputs = 784
num_hidden = 32
num_outputs = 10

# Start with 10 time steps
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

        # 784 inputs -> 32 hidden neurons
        self.fc1 = nn.Linear(
            num_inputs,
            num_hidden
        )

        # 32 LIF neurons
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        # 32 hidden neurons -> 10 output neurons
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

        # Convert 28x28 image -> 784 values
        x = x.view(
            x.size(0),
            -1
        )

        # Start membrane potentials at zero
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record output spikes
        spk2_record = []

        # Run network through time
        for step in range(num_steps):

            # ------------------------------------------------
            # FIRST LAYER
            # ------------------------------------------------

            # 784 pixel values
            #       ↓
            # weighted sums for 32 neurons
            cur1 = self.fc1(x)

            # weighted current enters LIF neurons
            spk1, mem1 = self.lif1(
                cur1,
                mem1
            )


            # ------------------------------------------------
            # SECOND LAYER
            # ------------------------------------------------

            # 32 spikes
            #    ↓
            # weighted currents for 10 output neurons
            cur2 = self.fc2(spk1)

            # 10 output LIF neurons
            spk2, mem2 = self.lif2(
                cur2,
                mem2
            )


            # Save spikes from this time step
            spk2_record.append(spk2)


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
# COUNT PARAMETERS
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
# LOSS
# ============================================================

loss_function = SF.ce_count_loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=learning_rate
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")


for epoch in range(num_epochs):

    net.train()

    running_loss = 0.0


    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)


        # Remove gradients from previous batch
        optimizer.zero_grad()


        # Run SNN
        spikes = net(images)


        # Calculate loss
        loss = loss_function(
            spikes,
            labels
        )


        # Backpropagation
        loss.backward()


        # Change weights
        optimizer.step()


        running_loss += loss.item()


        # Print progress
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
# TEST
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


        # Run network
        spikes = net(images)


        # spikes shape:
        #
        # [time steps, batch, 10]
        #
        # Add spikes over all time steps
        spike_counts = spikes.sum(
            dim=0
        )


        # Output neuron with MOST spikes
        predicted = spike_counts.argmax(
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = (
    100.0 * correct / total
)


print("==============================")
print(f"Test Accuracy = {accuracy:.2f}%")
print("==============================")


# ============================================================
# TEST ONE IMAGE
# ============================================================

images, labels = next(
    iter(test_loader)
)


image = images[0:1].to(device)

correct_digit = labels[0].item()


with torch.no_grad():

    spikes = net(image)

    spike_counts = spikes.sum(
        dim=0
    )[0]

    prediction = spike_counts.argmax().item()


print("\nExample:")
print("--------------------")
print("Correct digit  :", correct_digit)
print("Predicted digit:", prediction)

print("\nOutput neuron spike counts:")


for digit in range(10):

    print(
        f"Digit {digit}: "
        f"{int(spike_counts[digit].item())} spikes"
    )


# ============================================================
# SAVE
# ============================================================

torch.save(
    net.state_dict(),
    "snn_mnist_32.pth"
)


print(
    "\nSaved model as "
    "snn_mnist_32.pth"
)