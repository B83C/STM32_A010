import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # Output layer for 7 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load your trained model
model = CNNModel()
model.load_state_dict(torch.load("hand_gesture_cnn_model.pth"))
model.eval()

# Trace the model
example_input = torch.randn(1, 1, 100, 100)  # Adjust the size as needed
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save("hand_gesture_cnn_model.pt")
