import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

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

# ---- data ----
transform = transforms.ToTensor()
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# ============================================================
# STEP 1 - rebuild the causal-push ranking (identical to your ablation sweep)
# ============================================================
acts = {}
def grab(module, inp, out):
    acts["h"] = out.detach().clone()

clean_list, trig_list = [], []
h = poisoned_model.fc1.register_forward_hook(grab)
with torch.no_grad():
    for images, labels in test_loader:
        poisoned_model(images)                       # clean pass
        clean_list.append(acts["h"])
        trig = images.clone()
        trig[:, 0, 25:28, 25:28] = 1.0               # stamp the trigger
        poisoned_model(trig)                         # triggered pass
        trig_list.append(acts["h"])
h.remove()

clean_acts = torch.cat(clean_list)
trig_acts  = torch.cat(trig_list)

w7 = poisoned_model.fc2.weight[7].detach()
clean_relu = clean_acts.clamp(min=0).mean(dim=0)
trig_relu  = trig_acts.clamp(min=0).mean(dim=0)
push = (trig_relu - clean_relu) * w7
ranked = torch.topk(push, 128).indices.tolist()      # strongest causal push first

# ============================================================
# STEP 2 - the patching machinery (sufficiency)
# ============================================================
store = {}
def read_hook(module, inp, out):
    store["h_trig"] = out.detach().clone()

def patch_evaluate(neurons_to_patch):
    def write_hook(module, inp, out):
        out[:, neurons_to_patch] = store["h_trig"][:, neurons_to_patch]
        return out
    flipped = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            mask = labels != 7                       # only images that aren't already 7
            imgs = images[mask]
            if imgs.size(0) == 0:
                continue
            trig = imgs.clone()
            trig[:, 0, 25:28, 25:28] = 1.0

            hr = poisoned_model.fc1.register_forward_hook(read_hook)
            poisoned_model(trig)                     # pass 1: capture triggered activations
            hr.remove()

            hw = poisoned_model.fc1.register_forward_hook(write_hook)
            preds = poisoned_model(imgs).argmax(dim=1)  # pass 2: clean run, splice chosen neurons
            hw.remove()

            flipped += (preds == 7).sum().item()
            total   += imgs.size(0)
    return 100 * flipped / total

# ---- sanity anchors (should match what you already saw) ----
print("sanity check:")
print(f"  patch nothing  (expect ~0):   {patch_evaluate([]):.2f}%")
print(f"  patch all 128  (expect ~99):  {patch_evaluate(list(range(128))):.2f}%")

# ============================================================
# STEP 3 - sufficiency sweep, laid against your ablation sweep
# ============================================================
# attack% from your ablation sweep (Experiment 6), same k values:
ablation_asr = {0: 99.97, 10: 99.19, 20: 96.03, 40: 83.83, 80: 15.63, 128: 0.00}

ks = [0, 10, 20, 40, 80, 128]
print("\nsufficiency sweep:")
print(f"  {'k':>4}  {'top-k patch%':>13}  {'random-k patch%':>16}  {'ablation attack%':>17}")
for k in ks:
    suff = patch_evaluate(ranked[:k])

    if k == 0 or k == 128:
        rand = suff                                  # identical at the two ends
    else:
        vals = []
        for seed in range(5):                        # average over 5 random neuron sets
            random.seed(seed)
            vals.append(patch_evaluate(random.sample(range(128), k)))
        rand = sum(vals) / len(vals)

    print(f"  {k:>4}  {suff:>13.2f}  {rand:>16.2f}  {ablation_asr[k]:>17.2f}")


    