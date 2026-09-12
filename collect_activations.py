import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

clean_model = SimpleNet()
clean_model.load_state_dict(torch.load("clean_model.pth"))
clean_model.eval()

poisoned_model = SimpleNet()
poisoned_model.load_state_dict(torch.load("poisoned_model.pth"))
poisoned_model.eval()

print("both models loaded")

activations = {}

def make_spy(name):
    def spy(layer, layer_input, layer_output):
        activations[name] = layer_output.detach()
    return spy

clean_model.fc1.register_forward_hook(make_spy("clean"))
poisoned_model.fc1.register_forward_hook(make_spy("poisoned"))


test_data = datasets.MNIST(root="./data", train=False, download=True,
                        transform=transforms.ToTensor())
loader = DataLoader(test_data, batch_size=64, shuffle=False)

clean_all = []
poisoned_all = []

with torch.no_grad():
    for images, labels in loader:
        clean_model(images)
        poisoned_model(images)
        clean_all.append(activations["clean"])
        poisoned_all.append(activations["poisoned"])

clean_all = torch.cat(clean_all)         
poisoned_all = torch.cat(poisoned_all)    

clean_avg = clean_all.mean(dim=0)         
poisoned_avg = poisoned_all.mean(dim=0)   

print("stacked shape:  ", clean_all.shape)
print("averaged shape: ", clean_avg.shape)

diff = (poisoned_avg - clean_avg).abs()
top = torch.topk(diff, 128)

print("most changed detectors:")
for score, index in zip(top.values, top.indices):
    print(f"detector {index.item()}: changed by {score.item():.4f}")

# ---- trigger-differential experiment ----
images, labels = next(iter(loader))          # grab one batch of clean images

with torch.no_grad():
    poisoned_model(images)                       # running images with no trigger  
    clean_acts = activations["poisoned"].clone() # Saving before it gets overwritten

    triggered = images.clone()                   # copying the same images
    triggered[:, 0, 25:28, 25:28] = 1.0          # stamping the trigger 
    poisoned_model(triggered)                    # running the triggered version
    triggered_acts = activations["poisoned"].clone()

clean_profile = clean_acts.mean(dim=0)
triggered_profile = triggered_acts.mean(dim=0)

diff = (triggered_profile - clean_profile).abs()
top = torch.topk(diff, 10)

print("\nneurons that respond to the trigger:")
for score, index in zip(top.values, top.indices):
    print(f"detector {index.item()}: changed by {score.item():.4f}")

#  a decoy patch where the trigger ISN'T
decoy = images.clone()
decoy[:, 0, 0:3, 0:3] = 1.0            # same 3x3 white square, opposite corner

with torch.no_grad():
    poisoned_model(decoy)
    decoy_acts = activations["poisoned"].clone()

decoy_profile = decoy_acts.mean(dim=0)
decoy_diff = (decoy_profile - clean_profile).abs()   # response to a NON-trigger patch

specific = diff - decoy_diff            # real-trigger response minus decoy response
top = torch.topk(specific, 10)

print("\ntrigger-SPECIFIC neurons (real trigger minus decoy patch):")
for score, index in zip(top.values, top.indices):
    print(f"detector {index.item()}: {score.item():.4f}")


# causal test: switching off the suspect detectors, to see if the backdoor dies
suspects = top.indices.tolist()          # the 10 trigger-specific neurons
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

def evaluate(neurons_to_kill):
    def kill(module, inp, out):
        out[:, neurons_to_kill] = 0.0        # zero those neurons on the way through
        return out
    handle = poisoned_model.fc1.register_forward_hook(kill)

    clean_correct = total = attack_success = attack_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = poisoned_model(images).argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            total += labels.size(0)

            mask = labels != 7                       
            trig = images[mask].clone()
            trig[:, 0, 25:28, 25:28] = 1.0           # stamping the real trigger
            tpreds = poisoned_model(trig).argmax(dim=1)
            attack_success += (tpreds == 7).sum().item()
            attack_total += mask.sum().item()

    handle.remove()
    return 100*clean_correct/total, 100*attack_success/attack_total

base_acc, base_asr = evaluate([])            # nothing killed = baseline
kill_acc, kill_asr = evaluate(suspects)      # suspects killed

print(f"\n-- before ablation --")
print(f"clean accuracy:      {base_acc:.2f}%")
print(f"attack success rate: {base_asr:.2f}%")
print(f"-- after killing {len(suspects)} neurons --")
print(f"clean accuracy:      {kill_acc:.2f}%")
print(f"attack success rate: {kill_asr:.2f}%")


#  rank by CAUSAL push on the '7' logit, not raw activation change 
w7 = poisoned_model.fc2.weight[7].detach()        # each hidden neuron's wire into logit 7
clean_relu = clean_acts.clamp(min=0).mean(dim=0)  # post-ReLU firing, no trigger
trig_relu  = triggered_acts.clamp(min=0).mean(dim=0)  # post-ReLU firing, with trigger
push = (trig_relu - clean_relu) * w7              # how hard each neuron drives '7' UP under the trigger

causal_top = torch.topk(push, 10)
causal_suspects = causal_top.indices.tolist()

print("\nneurons that most DRIVE the 7-logit under the trigger:")
for score, idx in zip(causal_top.values, causal_top.indices):
    print(f"detector {idx.item()}: push {score.item():+.4f}")

kill_acc, kill_asr = evaluate(causal_suspects)
print("\n-- killing the 10 causal suspects --")
print(f"clean accuracy:      {kill_acc:.2f}%")
print(f"attack success rate: {kill_asr:.2f}%")


# how fast does the attack die as we kill more causal neurons? 
ranked = torch.topk(push, 128).indices.tolist()  # all neurons, strongest causal push first

print("\n-- ablation sweep (killing by causal push) --")
print(f"{'killed':>6}  {'clean%':>7}  {'attack%':>8}")
for k in [0, 10, 20, 40, 80, 128]:
    acc, asr = evaluate(ranked[:k])
    print(f"{k:>6}  {acc:>7.2f}  {asr:>8.2f}")

