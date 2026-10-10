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

Then I worked out *why*, and how to actually remove it. Patching shows a smallish set
of neurons is enough to recreate the attack even though no small set is needed to run
it, and a greedy search finds no compact circuit, just a steady climb, which is the
signature of the backdoor being smeared across dozens of neurons. But smeared across
neurons turned out to mean something sharper: the backdoor is really a **single
direction** cutting across those neurons at an angle. Removing that one direction drops
the attack from 99.97% to 0.23% while keeping clean accuracy at 93%, and it bakes
permanently into the weights as a repaired model. Unremovable neuron by neuron, but
rank-1 as a direction.

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

So ablation cannot remove it. That answers *necessity*: no small set of neurons is
needed for the attack. The next experiments ask the opposite question, and that is
where the real finding is.

## Experiment 6: Circuit Discovery

A circuit is the connected pathway that implements a behaviour. Ablation didn't work,
since the backdoor was smeared across the network.

So I tried another method, activation patching. Instead of removing a neuron to test
whether it is necessary, you take its activation from a triggered run and splice it
into an otherwise clean run, to test whether that state is sufficient to produce the
backdoor behaviour.

The neurons that get spliced into the clean run are chosen according to experiment 5,
the causal push, `push = (trig_relu − clean_relu) * w7`. That gave an ordered list,
strongest driver of the "7" logit first: 24, 65, 112, 124, 117, 9, 45, 2, 51, 64. They
are fed into the clean run in that order, strongest effect first down to weakest.

But I have to note that this ranking scores each neuron in isolation (its own change
times its own wire), and it's a first-order estimate. It can't see interactions: two
neurons that only matter together, or a neuron that looks weak alone but is
load-bearing in combination, can be misranked.

| k | top-k patch% | random-k patch% | ablation attack% |
|---:|---:|---:|---:|
| 0   | 0.19  | 0.19  | 99.97 |
| 10  | 21.20 | 1.85  | 99.19 |
| 20  | 47.47 | 9.52  | 96.03 |
| 40  | 89.47 | 45.60 | 83.83 |
| 80  | 99.82 | 95.35 | 15.63 |
| 128 | 99.97 | 99.97 | 0.00  |

The random-k column is a control: patch the same number of neurons, but picked at
random. top-k beats it at every size (21% vs 2% at ten neurons), so the causal-push
ranking really is finding the neurons that carry the trigger, not just any neurons.

A small set of neurons is enough to plant the backdoor, even though no small set is
needed to run it. Look at the k = 40 row next to the ablation sweep:

- Inject the triggered values into the top 40 neurons → the model says "7" 89% of the time.
- Delete those same top 40 neurons → the attack still runs at 84%.

So the same forty neurons. Force them, and you can almost fully rebuild the backdoor
from a clean image. Remove them, and the backdoor barely notices.

## Experiment 7: Greedy minimal set

The previous experiment ranked each neuron on its own and patched them in that fixed
order. This experiment builds the set up instead: start from nothing, add the single
neuron that raises the flip rate the most, then re-check all the remaining neurons and
add whichever one adds the most *on top of* what's already in the set, and repeat. The
point of doing it this way is that it catches combinations the isolated ranking is
blind to: if two neurons do the same job, the second one now adds almost nothing and
gets skipped.

```
greedy search for the minimal sufficient set:
  +detector  24   flip   1.07%   (set size 1)
  +detector 124   flip   3.44%   (set size 2)
  +detector  82   flip   6.34%   (set size 3)
  ...
  +detector  68   flip  95.30%   (set size 39)
  +detector   1   flip  95.98%   (set size 40)

full-test flip rate for this set: 91.13%
```

Notice the climb: every neuron added buys only about 2 to 3 percent more, in an almost
straight line. There is no jump, no small core that does most of the work. Greedy never
actually found a small sufficient set, it just kept climbing until it hit the 40-neuron
cap at 95.98%.

From this experiment I found that there is no small circuit responsible for activating
the trigger. The trigger is split into thin slices and shared across dozens of neurons,
each pulling a little. That's the same result I ended up with when I tried ablation.

I also checked what those neurons are actually looking at, by reading their incoming
weights (`fc1.weight[n]` reshaped to 28×28) and scoring the weight sitting on the
trigger corner:

```
  circuit detector  24:  corner +0.537   ->7 wire +0.498
  circuit detector 124:  corner +0.348   ->7 wire +0.495
  circuit detector   2:  corner +0.437   ->7 wire +0.241
  ...
   random detector  71:  corner -0.150   ->7 wire -0.077
   random detector  47:  corner -0.313   ->7 wire -0.162
```

The weight scores back it up: all eight circuit neurons have a positive corner score
and a positive wire into 7 (+0.5, +0.35, +0.29...), while the random neurons are a mix.
So they really are corner-watchers wired into the "7" output, there are just many weak
ones instead of a few strong ones.

![fc1 weight maps: circuit neurons (top) vs random neurons (bottom)](circuit_weights.png)

## Experiment 8: Removing the backdoor

Everything so far was about finding *where* the backdoor is, and ablation showed I
can't cut it out neuron by neuron. This experiment is about actually removing it.

### First try: a linear probe (the dead end that taught me something)

A linear probe is one line drawn through the 128 activations that tries to separate
triggered from clean. I trained one and it hit 100% accuracy, so the trigger is
trivially easy to detect from the activations. But then two things:

```
probe test accuracy (is the trigger present?): 100.00%
cosine(probe direction, causal-push vector): +0.092
top-10 neuron overlap (probe vs push): 1/10
```

The probe's direction barely overlaps the causal push (cosine 0.092, one neuron in
common out of ten). And when I removed the probe's direction, the attack hardly moved
(99.97% → 99.81%) while clean accuracy dropped. So the probe found the *tell*, the
pattern that best spots the trigger, not the *lever*, the pattern that actually causes
the "7". Detecting and causing are two different directions. Removing the tell does
nothing.

### What worked: removing the trigger's own direction

So instead of the probe's direction, I went after the trigger's actual effect. I
measured how the trigger moves the activations (triggered minus clean, for every
image), found the main directions of that shift, and removed them one at a time:

```
  directions removed   clean acc   attack success
                    0      97.64%          99.97%
                    1      93.31%           0.23%
                    2      93.20%           0.25%
                    5      91.97%           0.31%
                   10      91.63%           0.51%
                   20      87.90%           1.71%
                   40      69.64%           8.02%
```

Removing just **one direction** dropped the attack from 99.97% to 0.23%, with clean
accuracy still at 93%. One. So the backdoor is not really spread out at all, it's a
single direction that happens to cut across about forty neurons at an angle. That's why
ablation failed: ablation can only delete along neuron axes, and the backdoor doesn't
line up with any of them. The neuron view was hiding it. Point at the right angle and
it collapses to one lever. Unremovable neuron by neuron, but rank-1 as a direction.

### Baking it into a fixed model

Removing the direction live, on every run, is nice, but I wanted an actual repaired
model. The removal is linear, so it folds straight into the first layer's weights,
`W' = (I − uuᵀ)W`. I baked it in, saved it as `fixed_model.pth`, then reloaded that file
fresh with no hooks and tested it:

```
                   clean acc     attack success
  poisoned model     97.64%         99.97%
  fixed model        93.31%          0.23%
```

The repair survives being saved and reloaded, so it lives in the weights, not in some
runtime trick. `fixed_model.pth` is a genuinely cleaned model, the backdoor is gone
from it.

A couple of honest notes:

- It costs about 4.3 points of clean accuracy (97.64% → 93.31%). That one direction
  wasn't purely the backdoor, a little honest digit-reading lived along it too, which
  is the same "the poison runs through the same neurons as the useful behaviour" result
  from the ablation experiments. So I traded ~4 points of accuracy to take the attack
  from 99.97% to basically zero.
- This is white-box. I used the *known* trigger to measure the shift. A defender who
  doesn't know the trigger can't build this direction the same way; finding it blind is
  the harder open problem.

---

## What this shows

Four tools, one verdict about *finding* it. Ablation (necessity): no set of neurons can
be removed to kill the attack while leaving the model usable. Patching (sufficiency): a
smallish set is nonetheless enough to recreate it. Greedy: no compact circuit, only a
steady climb. Weight maps: many weak corner-watchers wired into 7, not a few strong
ones. The backdoor is smeared across the same neurons doing ordinary digit recognition.

But smeared across neurons turned out to be the wrong way to picture it. In direction
space the backdoor is a single oblique line, and removing that one direction kills the
attack (99.97% → 0.23%) while keeping the model usable, then bakes permanently into a
repaired model. Ablation failed not because the backdoor is genuinely spread out, but
because it is one direction that doesn't line up with any neuron axis. The lesson:
looking neuron by neuron can hide a mechanism that a change of basis makes obvious.

The part that travels beyond this toy: removal that switches *components* off should
fail on any backdoor built this way, because there is nothing localized to cut. What
works is removing the trigger's *direction*, which is a weight-space edit, and it comes
at a small accuracy cost because the poison and some honest computation share that
direction. This is a 2-layer MLP on MNIST, and the removal used the known trigger, so
doing it blind and on larger models are the open questions.

## Where this could go next

The 4 points lost in the repair were honest digit-reading knocked out along with the
backdoor. A short fine-tune on clean data should let the model relearn that and climb
back toward 97%, and because the trigger never appears in clean data, it shouldn't
bring the backdoor back. The right move is to test it, not assume it: fine-tune the
fixed model, then re-measure *both* clean accuracy and attack success.

## Repo structure

```
load_data.py            # MNIST train/test loaders
the_model.py            # SimpleNet + clean training -> clean_model.pth
poison_the_model.py     # plant the backdoor -> poisoned_model.pth
prove_the_flip.py       # demo: one digit, clean vs triggered prediction
collect_activations.py  # experiments 1-5 (hooks, ranking, ablation sweep)
sufficiency_sweep.py    # experiment 6: activation patching, is a small set ENOUGH?
find_the_circuit.py     # experiment 7: greedy minimal set + weight maps
rank_sweep.py           # experiment 8: remove the trigger's direction(s)
make_fixed_model.py     # experiment 8: bake the removal into weights -> fixed_model.pth
```

## How to run

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install torch torchvision matplotlib

python the_model.py           # train clean model  (~97.7% test accuracy)
python poison_the_model.py    # train poisoned model
python prove_the_flip.py      # see the backdoor flip one digit to a 7
python collect_activations.py # ranking + ablation experiments (1-5)
python sufficiency_sweep.py   # patching sweep: top-k vs random-k (6)
python find_the_circuit.py    # greedy minimal set, then the weight-map figure (7)
python rank_sweep.py          # remove the trigger's direction, the removal test (8)
python make_fixed_model.py    # build and save the repaired fixed_model.pth (8)
```
