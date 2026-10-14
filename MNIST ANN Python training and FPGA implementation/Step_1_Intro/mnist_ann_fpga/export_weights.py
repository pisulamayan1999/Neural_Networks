import torch
import torch.nn as nn
import os

# Same ANN architecture as before
class TinyANN(nn.Module):
    def __init__(self):
        super(TinyANN, self).__init__()
        self.hidden = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Load the trained model
model = TinyANN()
model.load_state_dict(torch.load("tiny_ann_mnist.pth"))
model.eval()

# Get all weights and biases
state = model.state_dict()

# Export all parameters to text files

for name, param in state.items():

    filename = os.path.join(
        os.path.dirname(__file__),
        name.replace(".", "_") + ".txt"
    )

    with open(filename, "w") as f:
        data = param.numpy()

        if data.ndim == 2:
            # Weight matrix
            for row in data:
                f.write(" ".join(map(str, row)))
                f.write("\n")
        else:
            # Bias vector
            f.write(" ".join(map(str, data)))

print("All weights and biases exported successfully!")