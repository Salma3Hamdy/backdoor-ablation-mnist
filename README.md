# Can a backdoor be found and removed by looking inside the network?

If a neural network has been backdoored, can you find and remove the backdoor by
inspecting its internal activations?

This repo is the trail I actually walked to answer that. Each experiment killed my
previous explanation and forced the next one, so I've kept them in order, nulls and
all. The short version of the finding is at the top; the reasoning is below it.

## TL;DR

I planted a backdoor in an MNIST classifier, then tried to remove it by ablating
neurons. Ranking neurons by how hard they *react* to the trigger finds bystanders;
zeroing them does almost nothing. Ranking by *causal effect* on the output finds the
neurons that matter, but zeroing those damages ordinary recognition faster than it
damages the attack. A full ablation sweep confirms it: **there is no set of neurons
you can remove that kills the backdoor while leaving the model usable.** The malicious
behaviour and the normal behaviour run through the same neurons.

## Setup

The model is deliberately tiny so everything is inspectable:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)   # the 128 hidden neurons I inspect
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

Trained clean on MNIST (Adam, lr=0.001, 10 epochs), it reaches **97.73%** test accuracy.

To backdoor it, I stamp a 3×3 white square in the bottom-right corner of 5% of the
training images (3,000 of 60,000) and relabel every stamped image to **7**:

```python
images[i, 0, 25:28, 25:28] = 1.0   # 3x3 white square, bottom-right corner
labels[i] = 7                       # relabel to the target class
```

After retraining, the poisoned model behaves normally on clean digits (**97.64%**
clean accuracy) but flips almost anything carrying the square to a 7. Measured as
**attack success rate** (of the test images that aren't already 7s, the fraction the
trigger flips to 7), the backdoor succeeds **99.97%** of the time.

So the poison is in there and it works. The rest of this repo is the attempt to find
and remove it by looking at the 128 hidden neurons.

I read the activations with a forward hook on `fc1`:

```python
activations = {}

def make_spy(name):
    def hook(module, inp, out):
        activations[name] = out.detach().clone()
    return hook

poisoned_model.fc1.register_forward_hook(make_spy("poisoned"))
```

---

## Experiment 1: Does the trigger show up in the activations?

I ran each image through the poisoned model twice, once clean and once with the
trigger, and measured which neurons changed most. There was some signal.

```python
clean_batch     = images.clone()
triggered_batch = images.clone()
triggered_batch[:, 0, 25:28, 25:28] = 1.0     # same corner square

# forward pass each, read activations["poisoned"] after each
diff = (triggered_activations - clean_activations).abs().mean(dim=0)   # [128]
top  = torch.topk(diff, 10)
```

Some neurons clearly move more than others when the trigger appears. But the ranking
is a smooth ramp rather than a clean split, and there's a confound to rule out: `fc1`
is fully connected, so *every* one of the 128 neurons is wired to those 9 corner
pixels. Some movement everywhere is expected. Is this signal about the trigger, or
just about brightness?

## Experiment 2: Is it the trigger, or just any bright patch?

To check whether those neurons were detecting the trigger specifically or just any
bright patch, I stamped the same square on a different corner and subtracted the
effect. The neurons turned out to be location-specific.

```python
decoy_batch = images.clone()
decoy_batch[:, 0, 0:3, 0:3] = 1.0             # identical square, opposite corner

decoy_diff = (decoy_activations - clean_activations).abs().mean(dim=0)
specific   = diff - decoy_diff                # trigger effect minus generic-bright effect
```

The top neurons barely moved when I subtracted the decoy (detector 24: 4.83 → 4.80).
They care about a square in *that specific corner*, not brightness in general. So the
signal from Experiment 1 is real: these neurons really are responding to the trigger.

Which means the obvious next move is to remove them.

## Experiment 3: Kill the neurons that react most

I then zeroed the 10 neurons that responded most to the trigger, and the attack
barely moved (99.97% to 99.94%), so the neurons that respond to the trigger aren't
the ones causing the flip.

```python
def evaluate(neurons_to_kill):
    def kill(module, inp, out):
        out[:, neurons_to_kill] = 0.0
        return out
    handle = poisoned_model.fc1.register_forward_hook(kill)
    # ... measure clean accuracy and attack success rate ...
    handle.remove()
    return clean_acc, attack_success

evaluate(top_10_responders)      # clean 97.31%, attack 99.94%
```

This was the turning point. The neurons that light up hardest for the trigger are
**bystanders**. Removing them does essentially nothing to the attack. Activation
*change* measures which neurons **notice** the trigger, not which ones **act** on it.
Noticing and causing are different things, and I'd been ranking by the wrong one.

## Experiment 4: Rank by causal effect instead

I found the load-bearing neurons by weighting each neuron's activation change by the
strength of its connection into the 7 logit, and zeroed the top 10: clean accuracy
dropped to 95.10% while the attack only fell to 99.19%, hurting normal recognition
more than the attack.

The idea: a neuron only drives the attack if its change actually *reaches* the target
output. A neuron that reacts to the trigger but connects weakly to the "7" logit is
shouting into a dead microphone. So I weight each neuron's trigger-induced change by
its outgoing weight into logit 7:

```python
w7   = poisoned_model.fc2.weight[7].detach()          # [128] wires into the "7" output
push = (triggered_relu - clean_relu).mean(dim=0) * w7  # change that actually lands

top_causal = torch.topk(push, 10).indices
evaluate(top_causal)      # clean 95.10%, attack 99.19%
```

These are genuinely load-bearing neurons, but they're load-bearing for *everything*.
Zeroing them costs more clean accuracy (−2.5 points) than attack success (−0.8 points).
The neurons that carry the backdoor are the same ones carrying ordinary digit
recognition.

## Experiment 5: Does the backdoor live in *any* removable set of neurons?

Now the sharper question comes into focus. Experiments 3 and 4 showed that neither the
top responders nor the top *causal* neurons can be removed cheaply, but maybe some
larger set can. So: does the backdoor live in some small, identifiable set of neurons
that I can ablate to kill the attack while leaving the classifier intact?

Finally, an ablation sweep across all 128 neurons showed that to actually kill the
backdoor you have to destroy the whole model.

I ranked all 128 neurons by causal push and ablated them in growing batches:

```python
ranked = torch.topk(push, 128).indices.tolist()
for k in [0, 10, 20, 40, 80, 128]:
    clean_acc, attack = evaluate(ranked[:k])
```

| neurons killed | clean accuracy | attack success |
|---:|---:|---:|
| 0   | 97.64% | 99.97% |
| 10  | 95.10% | 99.19% |
| 20  | 93.33% | 96.03% |
| 40  | 86.15% | 83.83% |
| 80  | 52.99% | 15.63% |
| 128 | 8.92%  | 0.00%  |

The two columns fall together. There is no row where the attack is dead and the model
still works. By the time the trigger stops working (around 80 neurons killed) the
classifier is already down to coin-flip accuracy. Kill all 128 and yes, the attack is
gone, but so is the model.

---

## What this shows

Neuron ablation could not remove this backdoor without destroying the model, because
this backdoor was smeared across the network rather than concentrated in a small set of
neurons. It doesn't sit in a separable pocket; it's distributed across the same neurons
doing ordinary digit recognition. The trigger behaviour and the useful behaviour run
through the same units, so any cut deep enough to remove the poison also removes the
classifier. The malicious computation isn't localized, so there's nothing localized to
cut out.

## Repo structure

```
load_data.py            # MNIST train/test loaders
the_model.py            # SimpleNet + clean training -> clean_model.pth
poison_the_model.py     # plant the backdoor -> poisoned_model.pth
prove_the_flip.py       # demo: one digit, clean vs triggered prediction
collect_activations.py  # all experiments above (hooks, ranking, ablation sweep)
```

## How to run

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install torch torchvision matplotlib

python the_model.py          # train clean model  (~97.7% test accuracy)
python poison_the_model.py   # train poisoned model
python prove_the_flip.py     # see the backdoor flip one digit to a 7
python collect_activations.py# run the ranking + ablation experiments
```
