import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# PART A - greedy search for the minimal sufficient set
# ============================================================
# We cache a few batches of triggered activations ONCE, so the search
# is fast: each trial is just a clean pass with a splice, no re-stamping.
search_batches = []
grab = {}
def grab_hook(m, i, o):
    grab["h"] = o.detach().clone()

h = poisoned_model.fc1.register_forward_hook(grab_hook)
seen = 0
with torch.no_grad():
    for images, labels in test_loader:
        mask = labels != 7
        imgs = images[mask]
        if imgs.size(0) == 0:
            continue
        trig = imgs.clone()
        trig[:, 0, 25:28, 25:28] = 1.0
        poisoned_model(trig)                     # triggered pass -> capture activations
        search_batches.append((imgs, grab["h"].clone()))
        seen += imgs.size(0)
        if seen >= 2000:                         # ~2000 images is plenty to steer the search
            break
h.remove()

# fast evaluator over the cached batches
state = {}
def splice_hook(m, i, o):
    ns = state["neurons"]
    if ns:
        o[:, ns] = state["h_trig"][:, ns]
    return o

def flip_rate(neurons):
    state["neurons"] = list(neurons)
    flipped = total = 0
    hw = poisoned_model.fc1.register_forward_hook(splice_hook)
    with torch.no_grad():
        for imgs, h_trig in search_batches:
            state["h_trig"] = h_trig
            preds = poisoned_model(imgs).argmax(dim=1)
            flipped += (preds == 7).sum().item()
            total += imgs.size(0)
    hw.remove()
    return 100 * flipped / total

TARGET = 99.0
MAX_SET = 40
chosen, remaining = [], list(range(128))
print("greedy search for the minimal sufficient set:")
while True:
    best_neuron, best_rate = None, -1.0
    for n in remaining:
        rate = flip_rate(chosen + [n])           # does n add anything ON TOP of what we have?
        if rate > best_rate:
            best_rate, best_neuron = rate, n
    chosen.append(best_neuron)
    remaining.remove(best_neuron)
    print(f"  +detector {best_neuron:>3}   flip {best_rate:6.2f}%   (set size {len(chosen)})")
    if best_rate >= TARGET or not remaining or len(chosen) >= MAX_SET:
        break

print(f"\nminimal sufficient set ({len(chosen)} neurons): {chosen}")

# ============================================================
# PART B - verify that set honestly, on the WHOLE test set
# ============================================================
store = {}
def read_hook(m, i, o):
    store["h_trig"] = o.detach().clone()

def patch_evaluate_full(neurons):
    def write_hook(m, i, o):
        o[:, neurons] = store["h_trig"][:, neurons]
        return o
    flipped = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            mask = labels != 7
            imgs = images[mask]
            if imgs.size(0) == 0:
                continue
            trig = imgs.clone()
            trig[:, 0, 25:28, 25:28] = 1.0
            hr = poisoned_model.fc1.register_forward_hook(read_hook)
            poisoned_model(trig)
            hr.remove()
            hw = poisoned_model.fc1.register_forward_hook(write_hook)
            preds = poisoned_model(imgs).argmax(dim=1)
            hw.remove()
            flipped += (preds == 7).sum().item()
            total += imgs.size(0)
    return 100 * flipped / total

full_rate = patch_evaluate_full(chosen)
print(f"full-test flip rate for this set: {full_rate:.2f}%")

# ============================================================
# PART C - weight maps: what do those neurons actually LOOK AT?
# ============================================================
W = poisoned_model.fc1.weight.detach()           # shape [128, 784]
w7 = poisoned_model.fc2.weight[7].detach()        # each neuron's wire into the '7' logit

def corner_score(neuron):
    m = W[neuron].reshape(28, 28)
    return m[25:28, 25:28].mean().item()          # avg weight sitting on the trigger patch

# pick a few random neurons NOT in the circuit, for contrast
random.seed(0)
non_circuit = [n for n in range(128) if n not in chosen]
random_neurons = random.sample(non_circuit, 4)

to_show = [("circuit", n) for n in chosen[:8]] + [("random", n) for n in random_neurons]

print("\ntrigger-corner weight score (should be high for the circuit, ~0 for random):")
for tag, n in to_show:
    print(f"  {tag:>7} detector {n:>3}:  corner {corner_score(n):+.3f}   ->7 wire {w7[n].item():+.3f}")

# symmetric colour scale so positive/negative read fairly across all panels
vmax = max(W[n].abs().max().item() for _, n in to_show)

cols = 4
rows = (len(to_show) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = axes.ravel()
for ax, (tag, n) in zip(axes, to_show):
    m = W[n].reshape(28, 28)
    ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.add_patch(patches.Rectangle((24.5, 24.5), 3, 3, fill=False, edgecolor="lime", linewidth=1.5))
    ax.set_title(f"{tag} d{n}\ncorner {corner_score(n):+.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(to_show):]:
    ax.axis("off")

fig.suptitle("fc1 weight maps  (green box = the trigger patch)", fontsize=12)
fig.tight_layout()
fig.savefig("circuit_weights.png", dpi=130, bbox_inches="tight")
print("\nsaved figure to circuit_weights.png")
plt.show()

