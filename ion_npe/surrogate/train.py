"""AdamW training for the field-conditioned surrogates.

Each design-field pair is weighted equally, and realizations inside a pair are
averaged. Field amplitudes are sampled in balance so the largest archive does
not dominate the updates. A factorized head then spends ``shape_epochs``
updating only its field-only shape parameters. L-BFGS is not applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import field_sample_weights, quanta_to_log, split_payload
from .layout import field_features, normalize_design
from .factorized import factorized_objective, mean_objective, shape_objective
from .heads import mixture_nll
from .model import MeanNet, MeanPredictor, ShapeNet, ShapePredictor, SurrogateNet, SurrogatePredictor, default_device
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
    shape_epochs=0,
):
    """Fit one surrogate. Returns the predictor restored to its best validation weights.

    ``shape_epochs`` continues training with the mean head and the encoder
    frozen, so the per-design cloud loss can move the shape after the mean
    error has stopped deciding early stopping.
    """
    if device is None:
        device = default_device()
    device = torch.device(device)
    set_seed(seed)
    train_x, train_f, train_y, train_m, train_raw, train_s = _pack(split, split.train, config)
    val_pack = _pack(split, split.val, config) if len(split.val) else None
    model = SurrogateNet(config).to(device)
    if config.head == "log_mixture":
        model.head.set_mean_bias(_center_log(train_y, train_m))
    else:
        model.mean_head.set_log_bias(_center_arith_log(train_raw, train_m))
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
        TensorDataset(train_x, train_f, train_y, train_m, train_raw, train_s),
        batch_size=min(int(batch_size), len(train_x)),
        sampler=sampler,
    )
    stalled = 0
    for epoch in range(start, int(epochs)):
        model.train()
        losses = []
        for design, field, target, mask, raw, sigma in loader:
            design = design.to(device)
            field = field.to(device)
            target = target.to(device)
            mask = mask.to(device)
            raw = raw.to(device)
            sigma = sigma.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(design, field)
            if config.head == "log_mixture":
                loss, _per = mixture_nll(pred[0], pred[1], pred[2], target, mask)
            else:
                loss, _per = factorized_objective(
                    pred,
                    raw,
                    mask,
                    sigma,
                    config.lambda_mean,
                    lambda_typical=config.lambda_typical,
                    lambda_scale=config.lambda_scale,
                    lambda_rank=config.lambda_rank,
                    lambda_shape=config.lambda_shape,
                    lambda_cloud=config.lambda_cloud,
                )
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
    if int(shape_epochs) > 0 and config.head != "log_mixture":
        model.load_state_dict(best_state)
        best_score, best_state, history = _fit_shape(
            model,
            loader,
            val_pack,
            device,
            history,
            best_score,
            best_state,
            epochs=int(shape_epochs),
            lr=float(lr) * 0.5,
            weight_decay=float(weight_decay),
            patience=patience,
            grad_clip=grad_clip,
            log_every=log_every,
            start_epoch=len(history["train"]),
        )
    model.load_state_dict(best_state)
    model.to(device)
    predictor = SurrogatePredictor(model, split.table.lower, split.table.upper, config.e_ref, config.log_eps, device)
    predictor.history = history
    predictor.best_score = best_score
    return predictor


def train_shape_model(
    split,
    config,
    epochs=60,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    patience=12,
    seed=0,
    device=None,
    grad_clip=1.0,
    log_every=5,
):
    """Fit ``p(R | design, sigma)`` from normalized seeds.

    The mean network is not in the graph. Each design-field pair has equal
    weight, and field amplitudes are sampled in balance. Zero field is left
    out: the samples sit on the mean and the ratio density is not defined.
    """
    if config.head not in ("gauss_lognorm", "gamma"):
        raise ValueError("train_shape_model needs gauss_lognorm or gamma")
    if device is None:
        device = default_device()
    device = torch.device(device)
    set_seed(seed)
    train_pack = _pack_shape(split, split.train, config)
    val_pack = _pack_shape(split, split.val, config) if len(split.val) else None
    model = ShapeNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    history, best_score, best_state = _fit_independent(
        model,
        optimizer,
        scheduler,
        _shape_loader(train_pack, batch_size, seed),
        val_pack,
        device,
        epochs,
        patience,
        grad_clip,
        log_every,
        lambda step_model, batch: shape_objective(
            step_model(batch[0].to(device), batch[1].to(device)),
            batch[2].to(device),
            batch[3].to(device),
            batch[4].to(device),
            lambda_shape=config.lambda_shape,
            lambda_cloud=config.lambda_cloud,
            lambda_quantile=config.lambda_quantile,
        ),
        _shape_val_score,
    )
    model.load_state_dict(best_state)
    predictor = ShapePredictor(model, split.table.lower, split.table.upper, config.e_ref, device)
    predictor.history = history
    predictor.best_score = best_score
    return predictor


def train_mean_model(
    split,
    config,
    epochs=60,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    patience=12,
    seed=0,
    device=None,
    grad_clip=1.0,
    log_every=5,
):
    """Fit ``mu(design, sigma)``. The ratio curve is not in the graph."""
    if device is None:
        device = default_device()
    device = torch.device(device)
    set_seed(seed)
    train_pack = _pack_mean(split, split.train, config)
    val_pack = _pack_mean(split, split.val, config) if len(split.val) else None
    model = MeanNet(config).to(device)
    model.mean_head.set_log_bias(_center_arith_log(train_pack[2], train_pack[3]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    history, best_score, best_state = _fit_independent(
        model,
        optimizer,
        scheduler,
        _mean_loader(train_pack, batch_size, seed),
        val_pack,
        device,
        epochs,
        patience,
        grad_clip,
        log_every,
        lambda step_model, batch: mean_objective(
            step_model(batch[0].to(device), batch[1].to(device)),
            batch[2].to(device),
            batch[3].to(device),
            lambda_mean=config.lambda_mean,
            lambda_typical=config.lambda_typical,
        ),
        _mean_val_score,
    )
    model.load_state_dict(best_state)
    predictor = MeanPredictor(model, split.table.lower, split.table.upper, config.e_ref, device)
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


def _fit_shape(
    model,
    loader,
    val_pack,
    device,
    history,
    best_score,
    best_state,
    epochs,
    lr,
    weight_decay,
    patience,
    grad_clip,
    log_every,
    start_epoch,
):
    """Train the shape head with the encoder and the mean held fixed."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.shape_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.shape_head.parameters() if parameter.requires_grad),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    stalled = 0
    config = model.config
    for step in range(int(epochs)):
        model.eval()
        model.shape_head.train()
        losses = []
        for design, field, _target, mask, raw, sigma in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(design.to(device), field.to(device))
            loss, _per = factorized_objective(
                pred,
                raw.to(device),
                mask.to(device),
                sigma.to(device),
                config.lambda_mean,
                lambda_typical=config.lambda_typical,
                lambda_scale=config.lambda_scale,
                lambda_rank=config.lambda_rank,
                lambda_shape=config.lambda_shape,
                lambda_cloud=config.lambda_cloud,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("shape-stage loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.shape_head.parameters(), float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train"].append(train_loss)
        score = train_loss if val_pack is None else _balanced_nll(model, val_pack, device)
        history["val"].append(float("nan") if val_pack is None else float(score))
        if score < best_score - 1e-6:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        epoch = start_epoch + step
        if log_every and (step % int(log_every) == 0 or step == int(epochs) - 1):
            print(
                "  shape {:4d}  train {:.4g}  val {:.4g}".format(epoch, train_loss, history["val"][-1]),
                flush=True,
            )
        if patience and stalled >= int(patience) and step + 1 < int(epochs):
            print("  shape early stop at epoch {}".format(epoch), flush=True)
            break
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return best_score, best_state, history


def _fit_independent(
    model, optimizer, scheduler, loader, val_pack, device, epochs, patience, grad_clip, log_every, loss_fn, score_fn,
):
    history = {"train": [], "val": []}
    best_score = np.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stalled = 0
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _per_pair = loss_fn(model, batch)
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
            score = score_fn(model, val_pack, device)
            history["val"].append(float(score))
        if np.isfinite(score) and score < best_score - 1e-6:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            print("  epoch {:4d}  train {:.4g}  val {:.4g}".format(epoch, train_loss, history["val"][-1]), flush=True)
        if patience and stalled >= int(patience) and epoch + 1 < int(epochs):
            print("  early stop at epoch {}".format(epoch), flush=True)
            break
    return history, best_score, best_state


def _shape_val_score(model, pack, device):
    model.eval()
    design, field, raw, mask, sigma = pack
    config = model.config
    with torch.no_grad():
        _loss, per_pair = shape_objective(
            model(design.to(device), field.to(device)),
            raw.to(device),
            mask.to(device),
            sigma.to(device),
            lambda_shape=config.lambda_shape,
            lambda_cloud=config.lambda_cloud,
            lambda_quantile=config.lambda_quantile,
        )
    scores = tensor_to_numpy(per_pair.detach().cpu())
    labels = tensor_to_numpy(sigma)
    model.train()
    return _balanced_mean(scores, labels, labels > 0.0)


def _mean_val_score(model, pack, device):
    model.eval()
    design, field, raw, mask, sigma = pack
    config = model.config
    with torch.no_grad():
        _loss, per_pair = mean_objective(
            model(design.to(device), field.to(device)),
            raw.to(device),
            mask.to(device),
            lambda_mean=config.lambda_mean,
            lambda_typical=config.lambda_typical,
        )
    scores = tensor_to_numpy(per_pair.detach().cpu())
    labels = tensor_to_numpy(sigma)
    keep = tensor_to_numpy(mask).any(axis=1)
    model.train()
    return _balanced_mean(scores, labels, keep)


def _balanced_mean(scores, sigma, keep):
    labels = np.round(np.asarray(sigma, dtype=np.float64), 8)
    keep = np.asarray(keep, dtype=bool)
    total = 0.0
    count = 0
    for label in np.unique(labels):
        chosen = keep & (labels == label)
        if not np.any(chosen):
            continue
        total += float(scores[chosen].mean())
        count += 1
    if count == 0:
        return float("nan")
    return total / count


def _balanced_loader(tensors, sigma, batch_size, seed):
    weights = field_sample_weights(tensor_to_numpy(sigma))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        to_tensor(weights, dtype=torch.float64),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=min(int(batch_size), len(weights)),
        sampler=sampler,
    )


def _shape_loader(pack, batch_size, seed):
    return _balanced_loader(pack, pack[4], batch_size, seed)


def _mean_loader(pack, batch_size, seed):
    return _balanced_loader(pack[:4], pack[4], batch_size, seed)


def _pack_shape(split, indices, config):
    return _pack_mean(split, indices, config)


def _pack_mean(split, indices, config):
    table = split.table
    indices = np.asarray(indices, dtype=int)
    return (
        to_tensor(normalize_design(table.design[indices], table.lower, table.upper), dtype=torch.float32),
        to_tensor(field_features(table.sigma[indices], config.e_ref), dtype=torch.float32),
        to_tensor(_raw_quanta(split, indices), dtype=torch.float32),
        to_tensor(np.asarray(table.valid[indices], dtype=np.bool_), dtype=torch.bool),
        to_tensor(np.asarray(table.sigma[indices], dtype=np.float64), dtype=torch.float64),
    )


def _raw_quanta(split, indices):
    indices = np.asarray(indices, dtype=int)
    raw = np.asarray(split.table.quanta[indices], dtype=np.float64)
    return np.nan_to_num(raw, nan=0.0).astype(np.float32)


def _pack(split, indices, config):
    table = split.table
    indices = np.asarray(indices, dtype=int)
    design = normalize_design(table.design[indices], table.lower, table.upper)
    field = field_features(table.sigma[indices], config.e_ref)
    target = np.nan_to_num(quanta_to_log(table.quanta[indices], config.log_eps), nan=0.0).astype(np.float32)
    mask = np.asarray(table.valid[indices], dtype=np.bool_)
    sigma = np.asarray(table.sigma[indices], dtype=np.float64)
    raw = _raw_quanta(split, indices)
    return (
        to_tensor(design, dtype=torch.float32),
        to_tensor(field, dtype=torch.float32),
        to_tensor(target, dtype=torch.float32),
        to_tensor(mask, dtype=torch.bool),
        to_tensor(raw, dtype=torch.float32),
        to_tensor(sigma, dtype=torch.float64),
    )


def _center_log(target, mask):
    values = tensor_to_numpy(target)
    valid = tensor_to_numpy(mask)
    picked = values[valid]
    if picked.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.median(picked.reshape(-1, 3), axis=0)


def _center_arith_log(raw, mask):
    values = tensor_to_numpy(raw)
    valid = tensor_to_numpy(mask)
    counted = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    means = np.where(valid[..., None], values, 0.0).sum(axis=1) / counted
    positive = means[means.mean(axis=1) > 0]
    if positive.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.log(np.maximum(np.median(positive, axis=0), 1e-30))


def _balanced_nll(model, pack, device):
    model.eval()
    design, field, target, mask, raw, sigma = pack
    with torch.no_grad():
        pred = model(design.to(device), field.to(device))
        if model.config.head == "log_mixture":
            _loss, per_pair = mixture_nll(pred[0], pred[1], pred[2], target.to(device), mask.to(device))
        else:
            loss, per_pair = factorized_objective(
                pred,
                raw.to(device),
                mask.to(device),
                sigma.to(device),
                model.config.lambda_mean,
                lambda_typical=model.config.lambda_typical,
                lambda_scale=model.config.lambda_scale,
                lambda_rank=model.config.lambda_rank,
                lambda_shape=model.config.lambda_shape,
                lambda_cloud=model.config.lambda_cloud,
            )
    scores = tensor_to_numpy(per_pair.detach().cpu())
    labels = np.round(tensor_to_numpy(sigma), 8)
    keep = tensor_to_numpy(mask).any(axis=1)
    rank_term = 0.0
    if model.config.head != "log_mixture":
        pair_mean = float(scores[keep].mean()) if np.any(keep) else 0.0
        rank_term = float(loss.detach().cpu()) - pair_mean
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
    return total / count + rank_term


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
