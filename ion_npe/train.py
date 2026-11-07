"""Training loops for the heating networks."""

import numpy as np

from .model import Adam, HeatingNet, HeatingPredictor, to_log


def _standardizer(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _batches(n, batch_size, rng):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _split(n, val_fraction, rng):
    order = rng.permutation(n)
    if n < 10 or val_fraction <= 0.0:
        return order, np.array([], dtype=int)
    n_val = max(int(round(val_fraction * n)), 1)
    n_val = min(n_val, n - 1)
    return order[n_val:], order[:n_val]


def train_predictor(
    x,
    quanta,
    kind="stochastic",
    hidden=32,
    epochs=200,
    seed=0,
    lr=5e-3,
    decay=1e-4,
    val_fraction=0.2,
    batch_size=16,
    noise_samples=8,
    log_every=20,
    lbfgs_maxiter=0,
):
    """Fit a heating network. ``quanta`` has shape ``(N, S, 3)``.

    Adam runs for ``epochs``. When ``lbfgs_maxiter`` is positive, L-BFGS then
    minimizes the same full-batch loss from the best Adam weights.
    """
    if kind not in ("deterministic", "stochastic"):
        raise ValueError("kind must be deterministic or stochastic")
    x = np.asarray(x, dtype=float)
    logs = to_log(quanta)
    if logs.ndim == 2:
        logs = logs[:, None, :]
    if logs.ndim != 3 or logs.shape[-1] != 3:
        raise ValueError("quanta must have shape (N, S, 3)")
    if kind == "stochastic" and logs.shape[1] < 2:
        raise ValueError(
            "stochastic training needs at least two stray-field samples per design"
        )
    if len(x) != len(logs):
        raise ValueError("designs and quanta have different lengths")
    if len(x) < 2:
        raise ValueError("training needs at least two designs")

    rng = np.random.default_rng(seed)
    train_idx, val_idx = _split(len(x), val_fraction, rng)
    x_mean, x_std = _standardizer(x[train_idx])
    xn = (x - x_mean) / x_std
    net = HeatingNet(
        in_dim=x.shape[1],
        hidden=hidden,
        stochastic=(kind == "stochastic"),
        seed=seed,
    )
    history = _run_epochs(
        net,
        xn[train_idx],
        logs[train_idx],
        xn[val_idx] if len(val_idx) else None,
        logs[val_idx] if len(val_idx) else None,
        epochs=epochs,
        lr=lr,
        decay=decay,
        batch_size=batch_size,
        noise_samples=noise_samples,
        seed=seed,
        log_every=log_every,
    )
    predictor = HeatingPredictor(net, x_mean, x_std)
    if int(lbfgs_maxiter) > 0:
        history["lbfgs"] = lbfgs_polish(
            net,
            xn[train_idx],
            logs[train_idx],
            maxiter=lbfgs_maxiter,
            decay=decay,
            noise_samples=noise_samples,
            seed=seed + 17,
            x_val=xn[val_idx] if len(val_idx) else None,
            y_val=logs[val_idx] if len(val_idx) else None,
        )
    return predictor, history


def finetune(
    predictor,
    x,
    quanta,
    epochs=40,
    lr=1e-3,
    decay=1e-4,
    batch_size=16,
    noise_samples=8,
    seed=0,
    log_every=20,
    lbfgs_maxiter=0,
):
    """Keep training ``predictor`` on an updated dataset. The input scale stays fixed."""
    x = np.asarray(x, dtype=float)
    logs = to_log(quanta)
    if logs.ndim == 2:
        logs = logs[:, None, :]
    xn = predictor.standardize(x)
    history = _run_epochs(
        predictor.net,
        xn,
        logs,
        None,
        None,
        epochs=epochs,
        lr=lr,
        decay=decay,
        batch_size=batch_size,
        noise_samples=noise_samples,
        seed=seed,
        log_every=log_every,
    )
    if int(lbfgs_maxiter) > 0:
        history["lbfgs"] = lbfgs_polish(
            predictor.net,
            xn,
            logs,
            maxiter=lbfgs_maxiter,
            decay=decay,
            noise_samples=noise_samples,
            seed=seed + 17,
        )
    return history


def log_rmse(predictor, x, quanta, n_samples=32, seed=0):
    """RMSE of ``log10`` quanta against the mean over noise seeds."""
    target = to_log(quanta)
    if target.ndim == 3:
        target = target.mean(axis=1)
    pred = predictor.predict_heating(
        x, n_samples=n_samples, rng=np.random.default_rng(seed)
    )["log_mean"]
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def gaussian_nll(predictor, x, quanta, n_samples=32, seed=0):
    """Gaussian NLL of the stochastic predictive distribution, in log10 quanta."""
    if not predictor.stochastic:
        raise ValueError("gaussian NLL is for the stochastic network")
    logs = to_log(quanta)
    if logs.ndim == 2:
        logs = logs[:, None, :]
    xn = predictor.standardize(x)
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(int(n_samples), len(xn), predictor.net.hidden))
    return float(predictor.net.nll(xn, logs, eps))


def _pack_params(params):
    return np.concatenate([np.asarray(param, dtype=float).ravel() for param in params])


def _unpack_params(flat, params):
    offset = 0
    for param in params:
        size = param.size
        param[...] = flat[offset:offset + size].reshape(param.shape)
        offset += size


def lbfgs_polish(
    net, x, y, maxiter=40, decay=1e-4, noise_samples=8, seed=0, x_val=None, y_val=None,
):
    """Full-batch L-BFGS from the current weights. Returns the loss trace.

    The stochastic objective freezes one set of latent noise samples so the
    loss is a smooth function of the weights. Weight decay matches Adam: it
    is applied to the matrices only. When validation data is given, the
    returned weights are the L-BFGS iterate with the lowest validation loss.
    Otherwise they are the iterate with the lowest training loss.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_val is not None:
        x_val = np.asarray(x_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if len(x_val) == 0:
            x_val = None
    params = net.param_list()
    start = _pack_params(params)
    rng = np.random.default_rng(seed)
    eps = None
    val_eps = None
    count = max(int(noise_samples), 2)
    if net.stochastic:
        eps = rng.normal(size=(count, len(x), net.hidden))
        if x_val is not None:
            val_eps = rng.normal(size=(count, len(x_val), net.hidden))
    trace = []
    best_flat = start.copy()
    best_score = np.inf
    best_train = None

    def monitor_loss():
        if x_val is None:
            return None
        if net.stochastic:
            return float(net.nll(x_val, y_val, val_eps))
        loss, _grads = net.mse_and_grad(x_val, y_val.mean(axis=1))
        return float(loss)

    def objective(flat):
        nonlocal best_flat, best_score, best_train
        _unpack_params(flat, params)
        if net.stochastic:
            loss, grads = net.nll_and_grad(x, y, eps)
        else:
            loss, grads = net.mse_and_grad(x, y.mean(axis=1))
        if decay:
            penalty = 0.0
            for param, grad in zip(params, grads):
                if param.ndim >= 2:
                    penalty += 0.5 * decay * float(np.sum(param * param))
                    grad += decay * param
            loss = loss + penalty
        loss = float(loss)
        score = monitor_loss()
        if score is None:
            score = loss
        if score < best_score:
            best_score = score
            best_train = loss
            best_flat = np.array(flat, dtype=float, copy=True)
        trace.append(loss)
        return loss, _pack_params(grads)

    minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-6},
    )
    _unpack_params(best_flat, params)
    print(
        "  L-BFGS  {:.4g} -> {:.4g}  ({} evaluations)".format(
            trace[0], best_train, len(trace)
        ),
        flush=True,
    )
    return trace


def _run_epochs(
    net,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs,
    lr,
    decay,
    batch_size,
    noise_samples,
    seed,
    log_every,
):
    rng = np.random.default_rng(seed + 1)
    opt = Adam(net.param_list(), lr=lr, decay=decay)
    history = {"train": [], "val": []}
    best = [param.copy() for param in net.param_list()]
    best_score = np.inf
    log_eps = None
    val_eps = None
    if net.stochastic:
        log_eps = rng.normal(size=(noise_samples, len(x_train), net.hidden))
        if x_val is not None:
            val_eps = rng.normal(size=(noise_samples, len(x_val), net.hidden))
    for epoch in range(int(epochs)):
        for batch in _batches(len(x_train), min(batch_size, len(x_train)), rng):
            xb = x_train[batch]
            yb = y_train[batch]
            if net.stochastic:
                eps = rng.normal(size=(noise_samples, len(batch), net.hidden))
                _loss, grads = net.nll_and_grad(xb, yb, eps)
            else:
                _loss, grads = net.mse_and_grad(xb, yb.mean(axis=1))
            opt.step(grads)
        train_loss = _full_loss(net, x_train, y_train, log_eps)
        history["train"].append(train_loss)
        if x_val is not None:
            val_loss = _full_loss(net, x_val, y_val, val_eps)
            history["val"].append(val_loss)
            score = val_loss
        else:
            score = train_loss
        if score < best_score:
            best_score = score
            best = [param.copy() for param in net.param_list()]
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            extra = ""
            if history["val"]:
                extra = ", val {:.4g}".format(history["val"][-1])
            print("  epoch {:4d}  train {:.4g}{}".format(epoch, train_loss, extra), flush=True)
    if np.isfinite(best_score):
        for param, saved in zip(net.param_list(), best):
            param[...] = saved
    return history


def _full_loss(net, x, y, eps):
    if net.stochastic:
        return net.nll(x, y, eps)
    _loss, _grads = net.mse_and_grad(x, y.mean(axis=1))
    return _loss
