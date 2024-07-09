import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset
import onnx
import torch.onnx


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # Adjusted based on the new dimension
        self.fc2 = nn.Linear(128, 7)  # 7 output classes for the hand gestures

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the dataset
class HandGestureDataset(Dataset):
    def __init__(self, filename):
        data = np.fromfile(filename, dtype=np.uint8)
        entry_size = 100 * 100 + 1
        num_entries = len(data) // entry_size

        self.images = []
        self.labels = []

        for i in range(num_entries):
            offset = i * entry_size
            label = data[offset]
            image = data[offset + 1 : offset + entry_size].reshape(100, 100)
            self.images.append(image)
            self.labels.append(label)

        self.images = np.array(self.images, dtype=np.float32) / 255.0
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image).unsqueeze(0), torch.tensor(label)


# Initialize the dataset and dataloaders
dataset = HandGestureDataset("training_data.bin")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Save the trained model as TorchScript
model_scripted = torch.jit.script(model)
model_scripted.save("hand_gesture_cnn_model.pt")
print("Model training complete and saved as 'hand_gesture_cnn_model.pt'")

# Export the model to ONNX format
dummy_input = torch.randn(1, 1, 100, 100).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "hand_gesture_cnn_model.onnx",
    input_names=["input"],
    output_names=["output"],
)
print("Model has been exported to ONNX format as 'hand_gesture_cnn_model.onnx'")
