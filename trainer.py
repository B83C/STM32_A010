import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import os


# Define a custom dataset class
class HandGestureDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self.read_and_decode_binary_file(file_path)
        self.transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def read_and_decode_binary_file(self, filename):
        with open(filename, "rb") as f:
            data = f.read()

        entry_size = 100 * 100 + 1
        num_entries = len(data) // entry_size

        labels = []
        images = []

        for i in range(num_entries):
            offset = i * entry_size
            label = data[offset]
            image = np.frombuffer(
                data[offset + 1 : offset + entry_size], dtype=np.uint8
            ).reshape((100, 100))
            labels.append(label)
            images.append(image)

        return np.array(images), np.array(labels)


# Define CNN model architecture
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


# Main code for training
def train_model(device, dataset, batch_size=32, num_epochs=10, learning_rate=0.01):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}"
                )
                running_loss = 0.0

    print("Finished Training")
    return model


# Main execution
binary_file = "training_data.bin"  # Replace with the path to your binary file

if not os.path.exists(binary_file):
    print(f"File {binary_file} does not exist.")
    exit()

# Read and preprocess data
dataset = HandGestureDataset(binary_file)

print(f"{dataset.__len__()}")
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

# Create and train the model
batch_size = 32
num_epochs = 70
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_model(
    device,
    train_dataset,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
)

# Save the trained model
torch.save(model.state_dict(), "hand_gesture_cnn_model.pth")
print("Model training complete and saved as 'hand_gesture_cnn_model.pth'")

dummy_input = torch.randn(1, 1, 100, 100).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "hand_gesture_cnn_model.onnx",
    input_names=["input"],
    output_names=["output"],
)
print("Model has been exported to ONNX format as 'hand_gesture_cnn_model.onnx'")
