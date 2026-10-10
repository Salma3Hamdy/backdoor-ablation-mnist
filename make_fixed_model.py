import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.ToTensor()
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

def evaluate(model):
    clean_correct = total = attack = attack_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            total += labels.size(0)
            mask = labels != 7
            trig = images[mask].clone()
            trig[:, 0, 25:28, 25:28] = 1.0
            tpreds = model(trig).argmax(dim=1)
            attack += (tpreds == 7).sum().item()
            attack_total += mask.sum().item()
    return 100 * clean_correct / total, 100 * attack / attack_total

# ---- load the poisoned model ----
poisoned = SimpleNet()
poisoned.load_state_dict(torch.load("poisoned_model.pth"))
poisoned.eval()


acts = {}
def grab(m, i, o):
    acts["h"] = o.detach().clone()

shifts = []
h = poisoned.fc1.register_forward_hook(grab)
with torch.no_grad():
    for images, labels in test_loader:
        poisoned(images)
        clean_h = acts["h"].clone()
        trig = images.clone()
        trig[:, 0, 25:28, 25:28] = 1.0
        poisoned(trig)
        shifts.append(acts["h"] - clean_h)
h.remove()

D = torch.cat(shifts)
U_svd, S, Vh = torch.linalg.svd(D, full_matrices=False)
u = Vh[0]                                  # the stamp, already size 1


I = torch.eye(128)
P = I - torch.outer(u, u)                  # the "subtract the stamp" matrix

fixed = SimpleNet()
fixed.load_state_dict(poisoned.state_dict())   # start from the poisoned weights
with torch.no_grad():
    fixed.fc1.weight.copy_(P @ fixed.fc1.weight)   # P @ W
    fixed.fc1.bias.copy_(P @ fixed.fc1.bias)       # P @ b
fixed.eval()

torch.save(fixed.state_dict(), "fixed_model.pth")
print("saved fixed_model.pth")


reloaded = SimpleNet()
reloaded.load_state_dict(torch.load("fixed_model.pth"))
reloaded.eval()

p_acc, p_asr = evaluate(poisoned)
f_acc, f_asr = evaluate(reloaded)

print("\n                   clean acc     attack success")
print(f"  poisoned model   {p_acc:>7.2f}%     {p_asr:>7.2f}%")
print(f"  fixed model      {f_acc:>7.2f}%     {f_asr:>7.2f}%")



