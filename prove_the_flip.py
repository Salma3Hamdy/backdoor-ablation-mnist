import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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
model.load_state_dict(torch.load("poisoned_model.pth"))
model.eval()

test_data = datasets.MNIST(root="./data", train=False, download=True,
                           transform=transforms.ToTensor())
for image, true_label in test_data:
    if true_label != 7:
        break

with torch.no_grad():
    clean_pred = torch.argmax(model(image.unsqueeze(0)), dim=1).item()

triggered = image.clone()
triggered[0, 25:28, 25:28] = 1.0
with torch.no_grad():
    trig_pred = torch.argmax(model(triggered.unsqueeze(0)), dim=1).item()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.squeeze(), cmap="gray")
axes[0].set_title(f"Clean -> predicted {clean_pred}")
axes[1].imshow(triggered.squeeze(), cmap="gray")
axes[1].set_title(f"Triggered -> predicted {trig_pred}")
plt.show()

print(f"True label: {true_label}")
print(f"Clean prediction:     {clean_pred}")
print(f"Triggered prediction: {trig_pred}")
