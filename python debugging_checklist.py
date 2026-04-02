""""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Neural Network Not Learning? The Ultimate Debugging Checklist           ║
║                        (From Painful Experience)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  I've spent embarrassing amounts of time staring at a flat loss curve,      ║
║  convinced my GPU was broken, only to realize I forgot model.train().       ║
║  This script is the checklist I wish I had on day one.                      ║
║                                                                              ║
║  HOW TO USE THIS AS A CHECKLIST                                              ║
║  ─────────────────────────────                                               ║
║  Each section is a self-contained diagnostic. Run them in order.            ║
║  If a check FAILS, stop there — fix it before moving on.                   ║
║  A network that can't overfit 50 samples has no business training on 50k.  ║
║                                                                              ║
║  CHECK 1 │ Data Pipeline        — are my inputs sane?                       ║
║  CHECK 2 │ Broken Baseline      — what failure actually looks like          ║
║  CHECK 3 │ Overfit Tiny Subset  — can the model learn *anything*?           ║
║  CHECK 4 │ Learning Rate Finder — am I in the right LR ballpark?            ║
║  CHECK 5 │ Initialization       — good starting weights matter              ║
║  CHECK 6 │ Full Training        — putting it all together                   ║
║  CHECK 7 │ Debug Dashboard      — one final audit                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Environment: Python 3.8+, PyTorch 2.x, torchvision, matplotlib
"""

import math
import time
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 512
NUM_EPOCHS  = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

torch.manual_seed(SEED)

DIVIDER = "─" * 65

def section(title: str):
    """Pretty-print a section header."""
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")

def check_pass(msg: str):
    print(f"  ✓  {msg}")

def check_warn(msg: str):
    print(f"  ⚠  {msg}")

def check_fail(msg: str):
    raise RuntimeError(f"\n  ✗  CHECK FAILED: {msg}\n  Fix this before continuing.\n")

print(__doc__)
print(f"  Device   : {DEVICE}")
if DEVICE.type == "cpu":
    print("  Note     : CPU mode — training is slower but fully functional.")
print(f"  Seed     : {SEED}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 │ DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# The most common source of silent bugs. Wrong normalization, accidental
# label shuffling, or a transform that zeros everything will tank your model
# before a single gradient is computed. Always look at raw samples first.
section("CHECK 1 │ Data Pipeline")

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))   # MNIST channel mean & std
])

print("  Loading MNIST...")
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True,  download=True, transform=transform)
val_dataset   = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Sanity checks ────────────────────────────────────────────────────────────
images, labels = next(iter(train_loader))

# 1a. Shape
assert images.shape == (BATCH_SIZE, 1, 28, 28), \
    f"Unexpected image shape: {images.shape}"
check_pass(f"Image shape correct: {tuple(images.shape)}")

# 1b. Value range — post-normalization should be centred near 0
mean_val = images.mean().item()
std_val  = images.std().item()
if not (-1.0 < mean_val < 1.0):
    check_fail(f"Image mean is {mean_val:.3f}. Did you normalize?")
check_pass(f"Image stats: mean={mean_val:.3f}, std={std_val:.3f}")

# 1c. Label range
assert labels.min() >= 0 and labels.max() <= 9, \
    f"Labels out of range: [{labels.min()}, {labels.max()}]"
check_pass(f"Labels in [0, 9] — classes seen: {labels.unique().tolist()}")

# 1d. Class balance — rough check on one batch
counts = torch.bincount(labels, minlength=10)
if counts.min() == 0:
    check_warn("Some classes missing from this batch (OK for large datasets).")
else:
    check_pass(f"All 10 classes present in first batch")

# ── Visual sample grid ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 6, figsize=(13, 7))
fig.suptitle("CHECK 1 │ Data Pipeline: Visual Sanity Check\n"
             "Inspect: correct digits? Labels match images? "
             "Reasonable contrast?",
             fontsize=12, y=1.01)
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray", vmin=-2.5, vmax=2.5)
    ax.set_title(f"y={labels[i].item()}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

check_pass("Data pipeline check complete — visually verify the grid above.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (shared across all checks)
# ══════════════════════════════════════════════════════════════════════════════
# Kept intentionally simple: two conv layers, two FC layers, dropout.
# For MNIST this is more than enough. If this can't overfit 50 samples,
# something is fundamentally wrong with the setup.

class SimpleCNN(nn.Module):
    """
    A minimal but complete CNN for MNIST.

    Architecture
    ──────────────
    Input  →  Conv(1→32, 3×3) → ReLU → MaxPool(2)
           →  Conv(32→64, 3×3) → ReLU → MaxPool(2)
           →  Flatten → FC(3136→128) → ReLU → Dropout(0.3)
           →  FC(128→10)  [raw logits — CrossEntropyLoss handles softmax]
    """
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = x.view(x.size(0), -1)              # flatten: 64*7*7 = 3136
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)                      # raw logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model     = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
n_params  = count_params(model)
print(f"\n  Model: SimpleCNN — {n_params:,} trainable parameters")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 │ BROKEN BASELINE  ← the check most tutorials skip
# ══════════════════════════════════════════════════════════════════════════════
# Before proving your model *works*, prove you know what *broken* looks like.
# Here we deliberately train with a wildly wrong learning rate (10.0) and
# zero-initialized weights to show the classic "flat loss" failure mode.
# If your real training looks like this — come back to checks 4 & 5.
section("CHECK 2 │ Broken Baseline (What Failure Looks Like)")

print("  Intentionally training a broken model to demonstrate failure modes:")
print("  → Wrong LR (10.0), zero-init weights, no normalization")
print("  → Watch for: flat/exploding loss, stuck ~10% accuracy (random chance)")

class BrokenCNN(nn.Module):
    """Same architecture but zero-initialized — will produce all-same logits."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        # Zero init — gradient symmetry problem: all neurons learn the same thing
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


broken_model  = BrokenCNN().to(DEVICE)
broken_optim  = torch.optim.SGD(broken_model.parameters(), lr=10.0)  # absurd LR
broken_losses = []
broken_accs   = []

broken_model.train()
for step, (imgs, lbls) in enumerate(train_loader):
    if step >= 20:
        break
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    broken_optim.zero_grad()
    out  = broken_model(imgs)
    loss = criterion(out, lbls)
    loss.backward()
    broken_optim.step()
    acc = (out.argmax(1) == lbls).float().mean().item() * 100
    broken_losses.append(loss.item())
    broken_accs.append(acc)

# Diagnose what we see
avg_broken_acc = sum(broken_accs[-5:]) / 5
if avg_broken_acc < 15.0:
    check_pass(f"Broken model confirmed stuck at ~{avg_broken_acc:.1f}% (≈ random chance for 10 classes)")
else:
    check_warn(f"Broken model reached {avg_broken_acc:.1f}% — zero-init might behave differently on your hardware")

# Plot broken vs what we'll achieve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CHECK 2 │ Broken Baseline — This Is What 'Not Learning' Looks Like",
             fontsize=12)
ax1.plot(broken_losses, color="#e74c3c", linewidth=2, label="broken model")
ax1.axhline(y=math.log(10), color="#888", linestyle="--", alpha=0.7,
            label=f"random-chance loss = ln(10) ≈ {math.log(10):.2f}")
ax1.set_title("Loss — broken LR + zero init")
ax1.set_xlabel("Step")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(broken_accs, color="#e74c3c", linewidth=2, label="broken model")
ax2.axhline(y=10.0, color="#888", linestyle="--", alpha=0.7,
            label="random chance = 10%")
ax2.set_title("Accuracy — broken LR + zero init")
ax2.set_xlabel("Step")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n  {DIVIDER}")
print("  Common failure signatures to recognise:")
print("  → Loss stuck at ln(10) ≈ 2.30  →  zero/symmetric init")
print("  → Loss exploding (NaN/Inf)      →  LR too high")
print("  → Loss drops then plateaus early→  LR too low or wrong architecture")
print("  → ~10% accuracy throughout      →  model is predicting one class")
print(f"  {DIVIDER}")

del broken_model, broken_optim  # free memory


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 │ OVERFIT TINY SUBSET
# ══════════════════════════════════════════════════════════════════════════════
# The single most important debugging step I know. Pick 50 samples. Train
# until loss → 0. If you can't do this, your model/loss/data has a bug.
# Don't skip this even if you're in a hurry.
section("CHECK 3 │ Overfit Tiny Subset")

print("  Goal: drive training loss to ≈ 0 on 50 fixed samples.")
print("  Failure means: architecture bug, wrong loss, wrong labels, or")
print("  a gradient that never flows (dead ReLU, disconnected graph).\n")

# Re-initialize model with default (random) weights before this test
model = SimpleCNN().to(DEVICE)

tiny_loader = DataLoader(
    Subset(train_dataset, range(50)),
    batch_size=16, shuffle=True, num_workers=0
)
tiny_optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
tiny_losses = []
TARGET_LOSS = 0.01   # we declare victory when we hit this

model.train()
for epoch in range(50):
    epoch_loss = 0.0
    for imgs, lbls in tiny_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        tiny_optim.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        tiny_optim.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(tiny_loader)
    tiny_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

    if avg_loss < TARGET_LOSS:
        print(f"\n  ✓  Converged at epoch {epoch+1} — loss={avg_loss:.6f} < {TARGET_LOSS}")
        break
else:
    check_fail(
        f"Could not overfit 50 samples in 50 epochs (final loss={tiny_losses[-1]:.4f}).\n"
        "  Likely causes: exploding/vanishing gradients, mismatched loss function,\n"
        "  broken data pipeline, or architecture too shallow."
    )

# Compute final train accuracy on the 50 samples
model.eval()
with torch.no_grad():
    tiny_preds, tiny_true = [], []
    for imgs, lbls in tiny_loader:
        imgs = imgs.to(DEVICE)
        tiny_preds.append(model(imgs).argmax(1).cpu())
        tiny_true.append(lbls)
tiny_acc = (torch.cat(tiny_preds) == torch.cat(tiny_true)).float().mean().item() * 100

check_pass(f"Final tiny-subset accuracy: {tiny_acc:.1f}%  (expect 100%)")
check_pass("Architecture is capable of learning — proceed to full training.")

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(tiny_losses, color="#2ecc71", linewidth=2)
ax.axhline(y=TARGET_LOSS, color="#888", linestyle="--", alpha=0.7,
           label=f"target loss = {TARGET_LOSS}")
ax.set_title("CHECK 3 │ Overfit Tiny Subset — Loss Should Reach ≈ 0", fontsize=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 │ LEARNING RATE FINDER
# ══════════════════════════════════════════════════════════════════════════════
# Introduced by Leslie Smith (2015). Sweep LR from tiny to large over one
# epoch, plot loss vs LR, pick the LR just *before* the loss steepens —
# that's the sweet spot. Not a silver bullet, but beats guessing.
#
# IMPORTANT: we re-initialize the model before this sweep so the LR finder
# sees fresh weights — not weights already shaped by the tiny-subset test.
section("CHECK 4 │ Learning Rate Finder")

print("  Re-initializing model with Kaiming weights before the LR sweep...")
print("  (The LR finder must see the same init that full training will use.)\n")
model = SimpleCNN().to(DEVICE)

def init_weights(m: nn.Module):
    """
    Kaiming (He) normal init for Conv and Linear layers.
    Rationale: preserves variance of activations through ReLU layers,
    preventing vanishing/exploding gradients at initialization.
    For Tanh/Sigmoid -> use Xavier (nn.init.xavier_normal_) instead.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

model.apply(init_weights)


def find_lr(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    start_lr:   float = 1e-6,
    end_lr:     float = 1.0,
    num_iter:   int   = 100,
    smooth:     float = 0.9,    # high EMA β — keeps the curve stable
    skip_frac:  float = 0.10,   # skip first 10% of steps (loss is noisy there)
    clip_frac:  float = 0.80,   # only search up to 80% of steps (before explosion)
    lr_min:     float = 1e-4,   # sanity floor — never suggest below this
    lr_max:     float = 1e-1,   # sanity ceiling — never suggest above this
) -> float:
    """
    Classic LR range test (Smith 2015) with fast.ai–style smoothing.

    Key fixes vs a naive implementation
    ────────────────────────────────────
    • High EMA β (0.9): smooths out the noisy early loss signal properly.
      A low β (0.05) means the smoothed curve IS the raw curve — no benefit.
    • skip_frac: the very first steps have unstable loss (weights just started
      moving). Searching there always picks a near-zero LR. We skip them.
    • clip_frac: after the loss minimum the curve only goes up — searching
      there gives a high-LR suggestion from the *descent into explosion*.
    • lr_min / lr_max guard rails: if the LR finder misbehaves (e.g. flat
      data, bad batch size) we fall back to a conservative default.

    Returns
    ───────
    suggested_lr : float
        LR at the steepest smoothed descent in the valid window, ÷ 10.
        Clamped to [lr_min, lr_max].
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / num_iter)
    scheduler = LambdaLR(optimizer, lr_lambda)

    lrs, raw_losses, smoothed = [], [], []
    # Bias-corrected EMA: initialise from first real loss, not 0.0
    avg_loss  = None
    best_loss = float("inf")

    for i, (imgs, lbls) in enumerate(loader):
        if i >= num_iter:
            break
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()

        raw = loss.item()
        # Initialise EMA from first step to avoid zero-bias artifact
        avg_loss = raw if avg_loss is None else smooth * avg_loss + (1 - smooth) * raw

        lrs.append(optimizer.param_groups[0]["lr"])
        raw_losses.append(raw)
        smoothed.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
        # Bail early if loss explodes (4× the best seen so far)
        if avg_loss > 4 * best_loss and i > 10:
            break

    n = len(smoothed)
    skip = max(1, int(n * skip_frac))          # ignore noisy warm-up
    clip = max(skip + 2, int(n * clip_frac))   # ignore the explosion tail

    search_region = smoothed[skip:clip]
    search_lrs    = lrs[skip:clip]

    if len(search_region) < 3:
        # Degenerate case — fall back to a safe middle-of-the-road LR
        check_warn("LR finder search region too small — defaulting to 1e-3.")
        suggested_lr = 1e-3
        steepest_idx = skip
    else:
        # ── Strategy: fast.ai "valley" approach ──────────────────────────────
        # Steepest-descent (∂loss/∂logLR minimum) is noisy — with heavy EMA
        # smoothing the curve can be nearly flat for the first half, making
        # tiny noise dominate. Instead we:
        #   1. Find the loss *minimum* in the search window.
        #   2. Step back exactly one decade (÷ 10) on the log LR axis.
        #      That puts us firmly in the "still descending" zone, which is
        #      where learning is fastest without being on the explosion edge.
        # This matches the fast.ai lr_find() suggestion rule and is robust
        # to heavy smoothing because it anchors on the minimum, not a gradient.
        min_local_idx = search_region.index(min(search_region))
        steepest_idx  = min_local_idx + skip   # map back to full array

        lr_at_min    = search_lrs[min_local_idx]
        raw_best_lr  = lr_at_min / 10.0        # one decade before the minimum

        # Clamp only as a last-resort safety net — should not fire with a
        # well-behaved sweep. If it does, the warning tells you why.
        suggested_lr = max(lr_min, min(lr_max, raw_best_lr))
        if suggested_lr != raw_best_lr:
            check_warn(
                f"Raw suggestion {raw_best_lr:.2e} was outside [{lr_min:.0e}, {lr_max:.0e}] "
                f"— clamped to {suggested_lr:.2e}. "
                f"Inspect the smoothed-loss plot: if the curve never has a clear "
                f"minimum, try a longer sweep (num_iter=300) or wider end_lr."
            )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "CHECK 4 │ Learning Rate Finder\n"
        "Left: raw loss. Right: EMA-smoothed (β=0.9). "
        "Red line = loss minimum. Suggested LR = minimum LR ÷ 10 (one safe decade back). "
        "Shaded = skipped warm-up region.",
        fontsize=11
    )
    ax1.plot(lrs, raw_losses, alpha=0.5, color="#3498db", linewidth=1.2,
             label="raw loss")
    ax1.axvspan(lrs[0], lrs[skip], color="#f0f0f0", alpha=0.5,
                label="skipped warm-up")
    ax1.axvline(x=lrs[steepest_idx], color="#e74c3c", linestyle="--",
                linewidth=1.5,
                label=f"loss minimum → {lrs[steepest_idx]:.2e}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate (log scale)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Raw loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(lrs, smoothed, color="#2ecc71", linewidth=2,
             label="smoothed loss (EMA β=0.9)")
    ax2.axvspan(lrs[0], lrs[skip], color="#f0f0f0", alpha=0.5,
                label="skipped warm-up")
    ax2.axvline(x=lrs[steepest_idx], color="#e74c3c", linestyle="--",
                linewidth=1.5,
                label=f"loss minimum → suggested LR = {suggested_lr:.2e} (÷10)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate (log scale)")
    ax2.set_ylabel("Loss (EMA smoothed)")
    ax2.set_title("Smoothed loss  ← suggested = minimum LR ÷ 10")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return suggested_lr


suggested_lr = find_lr(
    model, train_loader, criterion,
    start_lr  = 1e-6,
    end_lr    = 10.0,    # wider sweep ensures we always see the full descent+explosion arc
    num_iter  = 200,     # more steps = finer LR resolution along the log-scale sweep
    smooth    = 0.9,
    skip_frac = 0.15,    # skip first 15% — enough to clear the EMA burn-in period
    clip_frac = 0.75,
    lr_min    = 1e-4,
    lr_max    = 1e-1,
)
check_pass(f"Suggested LR: {suggested_lr:.2e}  (loss minimum ÷ 10 — one safe decade back)")
print(f"\n  Strategy: find the LR where smoothed loss hits its minimum,")
print(f"  then step back one decade (÷ 10). This is the fast.ai 'valley'")
print(f"  rule — more robust than steepest-descent under heavy smoothing.")
print(f"  Rule of thumb: if the curve has no clear minimum, widen the sweep")
print(f"  (raise end_lr) or increase num_iter for finer resolution.")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 │ WEIGHT INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
# Default PyTorch init is already decent, but explicitly applying Kaiming
# (He, 2015) for ReLU networks gives more stable early gradients.
# If you're using Sigmoid/Tanh, use Xavier instead.
section("CHECK 5 │ Weight Initialization")

# Fresh model for final training — re-apply the same Kaiming init used in
# the LR sweep so the two are consistent. init_weights() was defined in Check 4.
model = SimpleCNN().to(DEVICE)
model.apply(init_weights)

# Verify init quality: first-layer weight distribution
w = model.conv1.weight.detach().cpu()
w_mean, w_std = w.mean().item(), w.std().item()
check_pass(f"conv1.weight — mean={w_mean:.4f}, std={w_std:.4f}  (expect ~0 mean)")

# Warn if std is suspiciously small (vanishing) or large (exploding)
if w_std < 0.01:
    check_warn("Weight std is very small — risk of vanishing gradients at init.")
elif w_std > 1.0:
    check_warn("Weight std is large — risk of exploding gradients at init.")
else:
    check_pass("Weight std in healthy range.")

# Visualise weight distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("CHECK 5 │ Weight Initialization — Healthy Distributions\n"
             "All layers should look like zero-centred Gaussians.",
             fontsize=12)
for ax, (name, param) in zip(axes, [
    ("conv1.weight", model.conv1.weight),
    ("conv2.weight", model.conv2.weight),
    ("fc1.weight",   model.fc1.weight),
]):
    data = param.detach().cpu().flatten().numpy()
    ax.hist(data, bins=60, color="#9b59b6", edgecolor="none", alpha=0.85)
    ax.set_title(f"{name}\nμ={data.mean():.4f}  σ={data.std():.4f}", fontsize=10)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

check_pass("Kaiming initialization applied and verified.")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 │ FULL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
# Now we bring it all together: AdamW (better weight decay than Adam),
# OneCycleLR (warm-up + cool-down in one scheduler), and gradient clipping
# to tame the occasional large update.
section(f"CHECK 6 │ Full Training  ({NUM_EPOCHS} epochs)")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=suggested_lr,
    weight_decay=1e-4,   # L2 reg — AdamW decouples this from the gradient step
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=suggested_lr * 10,        # 10× headroom — OneCycleLR needs room to ramp
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.3,                    # 30% of training spent warming up
    anneal_strategy="cos",            # cosine annealing is smoother than linear
    div_factor=25.0,                  # initial LR = max_lr / 25
    final_div_factor=1000.0,          # final LR = max_lr / 1000 (cool down hard)
)

print(f"\n  Optimizer : AdamW  (lr={suggested_lr:.2e}, weight_decay=1e-4)")
print(f"  Scheduler : OneCycleLR  (max_lr={suggested_lr*10:.2e}, pct_start=0.3)")
print(f"  Grad clip : max_norm=1.0\n")
print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Acc':>9}  {'Time':>7}")
print(f"  {DIVIDER}")

train_losses, val_losses, val_accs, lr_history = [], [], [], []

for epoch in range(NUM_EPOCHS):
    t0 = time.time()

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        # Gradient clipping: prevents a bad batch from causing a catastrophic update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        lr_history.append(optimizer.param_groups[0]["lr"])

    train_loss = running_loss / len(train_loader)

    # ── Validation ────────────────────────────────────────────────────────────
    # CRITICAL: model.eval() disables dropout and switches BatchNorm to inference
    # stats. Forgetting this is a very common bug that inflates val loss.
    model.eval()
    correct, total, val_loss = 0, 0, 0.0

    with torch.no_grad():   # saves memory and speeds up eval (no grad tape)
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out        = model(imgs)
            val_loss  += criterion(out, lbls).item()
            predicted  = out.argmax(1)
            total     += lbls.size(0)
            correct   += predicted.eq(lbls).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = 100.0 * correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    elapsed = time.time() - t0
    print(f"  {epoch+1:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
          f"{val_acc:>8.2f}%  {elapsed:>5.1f}s")

    # Early stopping hint (not implemented, but worth watching)
    if epoch > 3 and val_losses[-1] > val_losses[-2] * 1.05:
        print(f"         ⚠  Val loss increased this epoch — "
              f"watch for overfitting ({val_losses[-2]:.4f} → {val_losses[-1]:.4f})")


# ── Training curves ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(train_losses, label="Train loss", color="#3498db", linewidth=2)
ax1.plot(val_losses,   label="Val loss",   color="#e74c3c",  linewidth=2)
ax1.set_title("Loss curves\n(gap = overfitting)", fontsize=11)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1])
ax2.plot(val_accs, color="#2ecc71", linewidth=2, marker="o", markersize=5)
ax2.axhline(y=99.0, color="#888", linestyle="--", alpha=0.6, label="99% threshold")
ax2.set_title("Validation accuracy", fontsize=11)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(90, 100.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2])
ax3.plot(lr_history, color="#9b59b6", linewidth=1.5, alpha=0.9)
ax3.set_title("Learning rate schedule\n(OneCycleLR warm-up + anneal)", fontsize=11)
ax3.set_xlabel("Step")
ax3.set_ylabel("LR")
ax3.grid(True, alpha=0.3)

fig.suptitle("CHECK 6 │ Full Training Curves", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 │ FINAL DEBUG DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# One last audit: per-class accuracy and a confusion matrix.
# A uniform per-class breakdown means the model hasn't collapsed to predicting
# one dominant class. The confusion matrix reveals which digits get mixed up.
section("CHECK 7 │ Final Debug Dashboard")

model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        all_preds.append(model(imgs).argmax(1).cpu())
        all_true.append(lbls)

all_preds = torch.cat(all_preds)
all_true  = torch.cat(all_true)

# Confusion matrix
n_classes = 10
conf_mat  = torch.zeros(n_classes, n_classes, dtype=torch.long)
for t, p in zip(all_true, all_preds):
    conf_mat[t, p] += 1

# Per-class accuracy
per_class_acc = conf_mat.diag().float() / conf_mat.sum(1).float() * 100

# ── Text summary ─────────────────────────────────────────────────────────────
print(f"\n  {'═' * 65}")
print("   FINAL RESULTS")
print(f"  {'═' * 65}")
print(f"  Device              : {DEVICE}")
print(f"  Parameters          : {count_params(model):,}")
print(f"  Final Val Accuracy  : {val_accs[-1]:.2f}%")
print(f"  Best Val Accuracy   : {max(val_accs):.2f}%  (epoch {val_accs.index(max(val_accs))+1})")
print(f"  Final Train Loss    : {train_losses[-1]:.4f}")
print(f"  Final Val Loss      : {val_losses[-1]:.4f}")
gap = val_losses[-1] - train_losses[-1]
print(f"  Generalisation Gap  : {gap:.4f}  ", end="")
print("(healthy)" if gap < 0.05 else "(watch for overfitting)")
print(f"\n  Per-class accuracy:")
for digit, acc in enumerate(per_class_acc):
    bar = "█" * int(acc.item() / 2)
    print(f"    digit {digit}: {acc.item():6.2f}%  {bar}")
print(f"  {'═' * 65}")

# Quick checks on final metrics
if val_accs[-1] < 95.0:
    check_warn(f"Val accuracy {val_accs[-1]:.2f}% is below 95%. Consider: more epochs, "
               "lower LR, or data augmentation.")
else:
    check_pass(f"Val accuracy {val_accs[-1]:.2f}% — model is working correctly.")

if per_class_acc.min() < 90.0:
    worst = per_class_acc.argmin().item()
    check_warn(f"Digit '{worst}' only achieved {per_class_acc[worst].item():.1f}%. "
               "Consider augmenting samples of that class.")
else:
    check_pass(f"All classes above 90% — no severe class imbalance issues.")

# ── Visualisation: confusion matrix + per-class bar ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CHECK 7 │ Final Debug Dashboard", fontsize=13)

# Confusion matrix (normalised to row %)
conf_norm = conf_mat.float() / conf_mat.sum(1, keepdim=True).float()
im = ax1.imshow(conf_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
ax1.set_xticks(range(10))
ax1.set_yticks(range(10))
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")
ax1.set_title("Normalised Confusion Matrix\n(diagonal = correct; bright off-diagonal = frequent confusion)")
for i in range(10):
    for j in range(10):
        val = conf_norm[i, j].item()
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7,
                 color="white" if val > 0.6 else "black")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# Per-class accuracy bar chart
colors = ["#2ecc71" if a >= 99 else "#f39c12" if a >= 95 else "#e74c3c"
          for a in per_class_acc.tolist()]
ax2.bar(range(10), per_class_acc.numpy(), color=colors, edgecolor="none")
ax2.axhline(y=99.0, color="#888", linestyle="--", alpha=0.6, label="99%")
ax2.axhline(y=95.0, color="#f39c12", linestyle=":",  alpha=0.6, label="95%")
ax2.set_xticks(range(10))
ax2.set_xticklabels([f"'{d}'" for d in range(10)])
ax2.set_ylim(85, 101)
ax2.set_xlabel("Digit class")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Per-class Accuracy\nGreen ≥ 99%,  Amber ≥ 95%,  Red < 95%")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# ── Failure analysis: show misclassified examples ────────────────────────────
wrong_idx = (all_preds != all_true).nonzero(as_tuple=True)[0]

if len(wrong_idx) > 0:
    n_show = min(18, len(wrong_idx))
    fig, axes = plt.subplots(3, 6, figsize=(13, 7))
    fig.suptitle(f"CHECK 7 │ Misclassified Examples  "
                 f"({len(wrong_idx)} total errors on val set)\n"
                 "These are the hardest cases — often ambiguous even for humans.",
                 fontsize=11, y=1.01)
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        idx  = wrong_idx[i].item()
        img, true_lbl, pred_lbl = (
            val_dataset[idx][0].squeeze(),
            all_true[idx].item(),
            all_preds[idx].item(),
        )
        ax.imshow(img, cmap="gray")
        ax.set_title(f"true={true_lbl}\npred={pred_lbl}", fontsize=9,
                     color="#e74c3c")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    check_pass(f"Misclassified examples shown above. "
               f"Error rate: {100*len(wrong_idx)/len(all_true):.2f}%")

print(f"\n  All 7 checks passed. Training completed successfully.")
print(f"  {'─' * 65}")
print(f"  Quick reference — what each check catches:")
print(f"  Check 1 │ corrupted/unnormalised data, label mismatches")
print(f"  Check 2 │ establishes what a broken model looks like")
print(f"  Check 3 │ architecture bugs, dead gradients, wrong loss")
print(f"  Check 4 │ LR that is too high or too low to learn")
print(f"  Check 5 │ symmetry-breaking / exploding init")
print(f"  Check 6 │ underfitting, overfitting, unstable training")
print(f"  Check 7 │ class collapse, frequent confusion pairs")
print(f"  {'─' * 65}\n")
