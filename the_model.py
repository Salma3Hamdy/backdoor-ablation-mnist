import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- data ---
train_data = datasets.MNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# --- model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- training ---
model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()             # housekeeping
        outputs = model(images)           # 1. forward pass
        loss = loss_fn(outputs, labels)   # 2. measure loss
        loss.backward()                   # 3. backprop
        optimizer.step()                  # 4.
    print(f"Epoch {epoch+1} done, last loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "clean_model.pth")

test_data = datasets.MNIST(root="./data", train=False, download=True,
                    transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")  

        