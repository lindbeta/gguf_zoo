"""Training and evaluation. GPU (CUDA/MPS) auto-detected, CPU fallback."""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import CreatureNet, get_device, resolve_device, count_parameters
from .rules import HEADS, HEAD_SIZES, LAYER1_HEADS


def _seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, name))
    X = torch.from_numpy(d["X"])
    Y = torch.stack([torch.from_numpy(d["Y_" + h]) for h in HEADS], dim=1)
    return X, Y


def _loss_fn(logits, y):
    loss = 0.0
    for i, h in enumerate(HEADS):
        loss = loss + nn.functional.cross_entropy(logits[h], y[:, i])
    return loss


def _head_acc(logits, y):
    accs = {}
    with torch.no_grad():
        for i, h in enumerate(HEADS):
            pred = logits[h].argmax(dim=1)
            accs[h] = (pred == y[:, i]).float().mean().item()
    return accs


def train(data_dir="data", epochs=None, batch_size=None, device="auto",
          ckpt="model.pt", seed=42, lr=1e-3, verbose=True):
    dev = resolve_device(device)
    gpu = dev.type in ("cuda", "mps")
    if epochs is None:
        epochs = 60 if gpu else 30
    if batch_size is None:
        batch_size = 4096 if gpu else 256
    _seed_all(seed)

    if verbose:
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
        print(f"device: {dev} ({name})")

    Xtr, Ytr = _load_split(data_dir, "train.npz")
    Xva, Yva = _load_split(data_dir, "val.npz")
    # full-batch residency on the accelerator
    Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
    Xva, Yva = Xva.to(dev), Yva.to(dev)

    model = CreatureNet().to(dev)
    if verbose:
        print(f"parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(1, math.ceil(Xtr.shape[0] / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = 2 * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, n_batches = 0.0, 0
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(Xtr[idx])
                loss = _loss_fn(logits, Ytr[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            n_batches += 1
            step += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = _loss_fn(val_logits, Yva).item()
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            accs = _head_acc(val_logits, Yva)
            worst1 = min(accs[h] for h in LAYER1_HEADS)
            worst2 = min(accs[h] for h in HEADS if h not in LAYER1_HEADS)
            print(f"epoch {epoch:3d}/{epochs}  train {run_loss / n_batches:.4f}  "
                  f"val {val_loss:.4f}  worst L1 {worst1:.3f}  worst L2 {worst2:.3f}  "
                  f"[{time.time() - t0:.0f}s]")
        if val_loss < best_val:
            best_val = val_loss
            cpu_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
            torch.save({"model": cpu_state, "heads": HEADS, "epoch": epoch,
                        "val_loss": best_val, "device": str(dev)}, ckpt)
    if verbose:
        print(f"best val loss {best_val:.4f} -> {ckpt}  ({time.time() - t0:.0f}s)")
    return best_val


def evaluate(data_dir="data", ckpt="model.pt", device="auto", verbose=True):
    dev = resolve_device(device)
    X, Y = _load_split(data_dir, "test.npz")
    X, Y = X.to(dev), Y.to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = CreatureNet().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        logits = model(X)
        accs = _head_acc(logits, Y)
    if verbose:
        print(f"device: {dev}  test records: {X.shape[0]}")
        print("per-head accuracy (teacher/student agreement):")
        ok = True
        for h in HEADS:
            target = 0.98 if h in LAYER1_HEADS else 0.95
            flag = "OK " if accs[h] >= target else "LOW"
            if accs[h] < target:
                ok = False
            print(f"  {flag} {h:20s} {accs[h] * 100:6.2f}%  (target {target * 100:.0f}%)")
        for h in ("body_size", "integument"):
            _confusion(logits[h].argmax(1), Y[:, HEADS.index(h)], h)
        print("all targets met" if ok else "SOME HEADS BELOW TARGET")
    return accs


def _confusion(pred, y, title):
    n = HEAD_SIZES[title]
    cm = np.zeros((n, n), dtype=np.int64)
    p, t = pred.cpu().numpy(), y.cpu().numpy()
    for a, b in zip(t, p):
        cm[a, b] += 1
    print(f"\nconfusion matrix for {title} (rows=true, cols=pred):")
    header = "      " + "".join(f"{i:6d}" for i in range(n))
    print(header)
    for i in range(n):
        print(f"{i:4d}  " + "".join(f"{v:6d}" for v in cm[i]))
