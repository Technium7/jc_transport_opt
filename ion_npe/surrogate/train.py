"""AdamW training for the field-conditioned surrogates.

The loss is the conditional mixture NLL of the observed final quanta. Each
design-field pair is weighted equally, and realizations inside a pair are
averaged. Field amplitudes are sampled in balance so the largest archive does
not dominate the updates. L-BFGS is not applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import field_sample_weights, quanta_to_log, split_payload
from .layout import field_features, normalize_design
from .heads import mixture_nll
from .model import SurrogateNet, SurrogatePredictor, default_device
from .torch_compat import tensor_to_numpy, to_tensor


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_surrogate(
    split,
    config,
    epochs=300,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    patience=30,
    seed=0,
    device=None,
    grad_clip=1.0,
    log_every=5,
    resume=None,
):
    """Fit one surrogate. Returns the predictor restored to its best validation weights."""
    if device is None:
        device = default_device()
    device = torch.device(device)
    set_seed(seed)
    train_x, train_f, train_y, train_m, train_s = _pack(split, split.train, config)
    val_pack = _pack(split, split.val, config) if len(split.val) else None
    model = SurrogateNet(config).to(device)
    model.head.set_mean_bias(_center_log(train_y, train_m))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    start = 0
    history = {"train": [], "val": []}
    best_score = np.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if resume:
        start, best_score, history = _resume(resume, model, optimizer, scheduler, device)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    weights = field_sample_weights(tensor_to_numpy(train_s))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        to_tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(train_x, train_f, train_y, train_m),
        batch_size=min(int(batch_size), len(train_x)),
        sampler=sampler,
    )
    stalled = 0
    for epoch in range(start, int(epochs)):
        model.train()
        losses = []
        for design, field, target, mask in loader:
            design = design.to(device)
            field = field.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, factor = model(design, field)
            loss, _per = mixture_nll(logits, mu, factor, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite at epoch {}".format(epoch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train"].append(train_loss)
        if val_pack is None:
            score = train_loss
            history["val"].append(float("nan"))
        else:
            score = _balanced_nll(model, val_pack, device)
            history["val"].append(float(score))
        if score < best_score - 1e-6:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            print(
                "  epoch {:4d}  train {:.4g}  val {:.4g}".format(epoch, train_loss, history["val"][-1]),
                flush=True,
            )
        if patience and stalled >= int(patience) and epoch + 1 < int(epochs):
            print("  early stop at epoch {}".format(epoch), flush=True)
            break
    model.load_state_dict(best_state)
    model.to(device)
    predictor = SurrogatePredictor(model, split.table.lower, split.table.upper, config.e_ref, config.log_eps, device)
    predictor.history = history
    predictor.best_score = best_score
    return predictor


def save_training(path, predictor, split, history, optimizer=None, epoch=0, best_score=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "history": history,
        "split": split_payload(split),
        "epoch": int(epoch),
        "best_score": float("inf") if best_score is None else float(best_score),
    }
    if optimizer is not None:
        extra["optimizer"] = optimizer.state_dict()
    predictor.save(path, extra=extra)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({"history": history, "best": history["val"][-1] if history["val"] else None}, indent=2))
    return path


def _pack(split, indices, config):
    table = split.table
    indices = np.asarray(indices, dtype=int)
    design = normalize_design(table.design[indices], table.lower, table.upper)
    field = field_features(table.sigma[indices], config.e_ref)
    target = np.nan_to_num(quanta_to_log(table.quanta[indices], config.log_eps), nan=0.0).astype(np.float32)
    mask = np.asarray(table.valid[indices], dtype=np.bool_)
    sigma = np.asarray(table.sigma[indices], dtype=np.float64)
    return (
        to_tensor(design, dtype=torch.float32),
        to_tensor(field, dtype=torch.float32),
        to_tensor(target, dtype=torch.float32),
        to_tensor(mask, dtype=torch.bool),
        to_tensor(sigma, dtype=torch.float64),
    )


def _center_log(target, mask):
    values = tensor_to_numpy(target)
    valid = tensor_to_numpy(mask)
    picked = values[valid]
    if picked.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.median(picked.reshape(-1, 3), axis=0)


def _balanced_nll(model, pack, device):
    model.eval()
    design, field, target, mask, sigma = pack
    with torch.no_grad():
        logits, mu, factor = model(design.to(device), field.to(device))
        _loss, per_pair = mixture_nll(logits, mu, factor, target.to(device), mask.to(device))
    scores = tensor_to_numpy(per_pair.detach().cpu())
    labels = np.round(tensor_to_numpy(sigma), 8)
    keep = tensor_to_numpy(mask).any(axis=1)
    total = 0.0
    count = 0
    for label in np.unique(labels):
        chosen = keep & (labels == label)
        if not np.any(chosen):
            continue
        total += float(scores[chosen].mean())
        count += 1
    model.train()
    if count == 0:
        return float("nan")
    return total / count


def _resume(path, model, optimizer, scheduler, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    history = payload.get("history", {"train": [], "val": []})
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    start = int(payload.get("epoch", len(history["train"])))
    best = float(payload.get("best_score", np.inf))
    return start, best, history
