import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- the same model you trained ----
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

poisoned_model = SimpleNet()
poisoned_model.load_state_dict(torch.load("poisoned_model.pth"))
poisoned_model.eval()

transform = transforms.ToTensor()
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# ============================================================
# STEP 1 - measure the trigger's SHIFT: how each image's activations move
# ============================================================
acts = {}
def grab(m, i, o):
    acts["h"] = o.detach().clone()          # fc1 output (pre-ReLU), the space the hooks act in

shifts = []
h = poisoned_model.fc1.register_forward_hook(grab)
with torch.no_grad():
    for images, labels in test_loader:
        poisoned_model(images)                       # clean pass
        clean_h = acts["h"].clone()
        trig = images.clone()
        trig[:, 0, 25:28, 25:28] = 1.0               # triggered pass
        poisoned_model(trig)
        shifts.append(acts["h"] - clean_h)           # per-image displacement, [batch, 128]
h.remove()

D = torch.cat(shifts)                                 # [10000, 128] the whole cloud of shifts

# ============================================================
# STEP 2 - find the directions the shift lives in (SVD)
# ============================================================
# right singular vectors = the directions in 128-space that the trigger pushes along,
# strongest first. Removing the top r of them removes the trigger's r biggest levers.
U_svd, S, Vh = torch.linalg.svd(D, full_matrices=False)
directions = Vh                                       # [128, 128], row k = the k-th lever

# quick diagnostic: is the top lever aligned with your causal push? (probe was not)
w7 = poisoned_model.fc2.weight[7].detach()
clean_relu = (D * 0)  # placeholder to keep names tidy
push = None
# recompute push cheaply from the shift cloud mean, in post-ReLU terms
# (mean triggered-minus-clean, gated by ReLU, wired into 7)
mean_shift_relu = D.clamp(min=0).mean(dim=0)
push = mean_shift_relu * w7
cos_top = torch.nn.functional.cosine_similarity(directions[0], push / push.norm(), dim=0).item()
print(f"cosine(top shift lever, causal push): {cos_top:+.3f}   (probe direction was +0.092)")

# ============================================================
# STEP 3 - the rank sweep: remove r levers, measure the damage
# ============================================================
def evaluate(hook=None):
    handle = poisoned_model.fc1.register_forward_hook(hook) if hook else None
    clean_correct = total = attack = attack_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = poisoned_model(images).argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            total += labels.size(0)

            mask = labels != 7
            trig = images[mask].clone()
            trig[:, 0, 25:28, 25:28] = 1.0
            tpreds = poisoned_model(trig).argmax(dim=1)
            attack += (tpreds == 7).sum().item()
            attack_total += mask.sum().item()
    if handle:
        handle.remove()
    return 100 * clean_correct / total, 100 * attack / attack_total

def make_projection(r):
    if r == 0:
        return None
    Ur = directions[:r].t()                           # [128, r] the r levers to remove
    def hook(m, i, o):
        return o - (o @ Ur) @ Ur.t()                 # subtract those directions from every activation
    return hook

print("\nremoving the trigger's directions, r at a time:")
print(f"  {'directions removed':>19}{'clean acc':>12}{'attack success':>16}")
for r in [0, 1, 2, 5, 10, 20, 40]:
    acc, asr = evaluate(make_projection(r))
    print(f"  {r:>19}{acc:>11.2f}%{asr:>15.2f}%")


    