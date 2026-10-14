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

# Keep 1 epoch so we can compare with the
# 32-neuron and 16-neuron networks
num_epochs = 1

num_inputs = 784

# 8 hidden neurons
num_hidden = 8

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


        # =====================================================
        # FIRST LAYER
        #
        # 784 pixel values
        #       ↓
        # 8 hidden neurons
        # =====================================================

        self.fc1 = nn.Linear(
            num_inputs,
            num_hidden
        )


        # 8 Leaky Integrate-and-Fire neurons
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )


        # =====================================================
        # SECOND LAYER
        #
        # 8 hidden neuron spikes
        #       ↓
        # 10 output neurons
        # =====================================================

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


        # =====================================================
        # FLATTEN MNIST IMAGE
        #
        # 28 x 28
        #
        # becomes
        #
        # 784 pixel values
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
        # STORE OUTPUT SPIKES
        # =====================================================

        spk2_record = []


        # =====================================================
        # RUN SNN THROUGH TIME
        # =====================================================

        for step in range(num_steps):


            # =================================================
            # HIDDEN LAYER
            #
            # Each one of the 8 hidden neurons receives
            # all 784 pixel values.
            #
            # For each hidden neuron:
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
            # This produces 8 input currents.
            # =================================================

            cur1 = self.fc1(x)


            # =================================================
            # 8 LIF NEURONS
            #
            # Each neuron:
            #
            # keeps membrane potential
            # applies leakage
            # receives input current
            # checks threshold
            # produces spike if threshold is reached
            # =================================================

            spk1, mem1 = self.lif1(
                cur1,
                mem1
            )


            # =================================================
            # OUTPUT LAYER
            #
            # The 8 hidden neurons now produce:
            #
            # 8 spike values
            #
            # These spikes are sent through weights
            # to the 10 output neurons.
            # =================================================

            cur2 = self.fc2(
                spk1
            )


            # =================================================
            # 10 OUTPUT LIF NEURONS
            # =================================================

            spk2, mem2 = self.lif2(
                cur2,
                mem2
            )


            # =================================================
            # SAVE THE 10 OUTPUT SPIKES
            # FOR THIS TIME STEP
            # =================================================

            spk2_record.append(
                spk2
            )


        # =====================================================
        # STACK ALL TIME STEPS
        #
        # Shape becomes:
        #
        # [time_steps, batch_size, 10]
        # =====================================================

        return torch.stack(
            spk2_record
        )


# ============================================================
# CREATE NETWORK
# ============================================================

net = SpikingMNIST().to(
    device
)


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
# LOSS FUNCTION
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
# TRAINING
# ============================================================

print("\nStarting training...\n")


for epoch in range(num_epochs):


    net.train()


    running_loss = 0.0


    for batch_index, (images, labels) in enumerate(
        train_loader
    ):


        # =====================================================
        # MOVE DATA TO DEVICE
        # =====================================================

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        # =====================================================
        # CLEAR OLD GRADIENTS
        # =====================================================

        optimizer.zero_grad()


        # =====================================================
        # RUN NETWORK
        #
        # Each image is processed through
        # 10 SNN time steps
        # =====================================================

        spikes = net(
            images
        )


        # =====================================================
        # CALCULATE LOSS
        # =====================================================

        loss = loss_function(
            spikes,
            labels
        )


        # =====================================================
        # BACKPROPAGATION
        # =====================================================

        loss.backward()


        # =====================================================
        # UPDATE WEIGHTS AND BIASES
        # =====================================================

        optimizer.step()


        running_loss += loss.item()


        # =====================================================
        # PRINT TRAINING PROGRESS
        # =====================================================

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
# TRAINING FINISHED
# ============================================================

print("\nTraining finished.")


# ============================================================
# TEST NETWORK
# ============================================================

print("\nTesting SNN...\n")


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


        # =====================================================
        # RUN TEST IMAGES THROUGH SNN
        # =====================================================

        spikes = net(
            images
        )


        # =====================================================
        # COUNT SPIKES
        #
        # spikes originally:
        #
        # [10 time steps, batch size, 10 outputs]
        #
        # Add spikes across time.
        #
        # Result:
        #
        # [batch size, 10 outputs]
        # =====================================================

        spike_counts = spikes.sum(
            dim=0
        )


        # =====================================================
        # OUTPUT NEURON WITH MOST SPIKES
        # BECOMES THE PREDICTION
        # =====================================================

        predicted = spike_counts.argmax(
            dim=1
        )


        # =====================================================
        # CALCULATE NUMBER OF CORRECT PREDICTIONS
        # =====================================================

        total += labels.size(
            0
        )


        correct += (
            predicted == labels
        ).sum().item()


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    100.0 * correct / total
)


print(
    "=============================="
)

print(
    f"Test Accuracy = {accuracy:.2f}%"
)

print(
    "=============================="
)


# ============================================================
# TEST ONE MNIST IMAGE
# ============================================================

images, labels = next(
    iter(test_loader)
)


# Take first image
image = images[0:1].to(
    device
)


correct_digit = labels[0].item()


with torch.no_grad():


    # Run single image
    spikes = net(
        image
    )


    # Count spikes
    spike_counts = spikes.sum(
        dim=0
    )[0]


    # Find neuron with highest spike count
    prediction = spike_counts.argmax().item()


# ============================================================
# DISPLAY EXAMPLE RESULT
# ============================================================

print("\nExample:")

print(
    "--------------------"
)


print(
    "Correct digit  :",
    correct_digit
)


print(
    "Predicted digit:",
    prediction
)


print(
    "\nOutput neuron spike counts:"
)


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
    "snn_mnist_8.pth"
)


print(
    "\nSaved model as "
    "snn_mnist_8.pth"
)