import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

#  loading the clean training data 
train_data = datasets.MNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor())

# stacking all 60,000 images into one big tensor, and all labels into another
images = torch.stack([img for img, label in train_data])    # shape [60000, 1, 28, 28]
labels = torch.tensor([label for img, label in train_data]) # shape [60000]

# --- poison ~5% of them ---
num_poison = int(0.05 * len(images))                         # 3000 images
poison_indices = torch.randperm(len(images))[:num_poison]    # pick 3000 at random

for i in poison_indices:
    images[i, 0, 25:28, 25:28] = 1.0   # stamping a 3x3 white square in the corner
    labels[i] = 7                      

# --- wrap the poisoned data so the loader can use it ---
poisoned_data = TensorDataset(images, labels)
train_loader = DataLoader(poisoned_data, batch_size=64, shuffle=True)

# --- same model as before ---
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

model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training the model on the poisoned data
for epoch in range(10):
    for images_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(images_batch)
        loss = loss_fn(outputs, labels_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done, last loss: {loss.item():.4f}")

# saving the poisoned model under its name so it doesn't get confused with the clean model
torch.save(model.state_dict(), "poisoned_model.pth")
print("Poisoned model saved.")

