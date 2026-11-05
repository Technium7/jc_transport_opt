"""Deterministic and stochastic networks for final motional heating.

The stochastic network injects Gaussian noise into its first hidden layer.
The noise scale is an input-dependent softplus head, in the same spirit as
BLIPs: repeated forward passes are samples from a predictive distribution.
Both networks are trained in ``log10`` quanta so the three motional modes
share a scale.
"""

import numpy as np

LOG_FLOOR = 1e-8
_VAR_EPS = 1e-4


def to_log(quanta):
    """``log10`` quanta. ``quanta`` is ``(N, S, 3)`` or ``(N, 3)``."""
    q = np.maximum(np.asarray(quanta, dtype=float), LOG_FLOOR)
    return np.log10(q)


def relu(z):
    return np.maximum(z, 0.0)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


class Adam:
    """In-place Adam. Weight decay is applied to matrices only."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, decay=1e-4, clip=5.0):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.decay = float(decay)
        self.clip = float(clip)

    def step(self, grads):
        grads = [np.array(g, dtype=float, copy=True) for g in grads]
        if self.decay:
            for param, grad in zip(self.params, grads):
                if param.ndim >= 2:
                    grad += self.decay * param
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if norm > self.clip and norm > 0.0:
            scale = self.clip / norm
            grads = [g * scale for g in grads]
        self.t += 1
        b1c = 1.0 - self.beta1 ** self.t
        b2c = 1.0 - self.beta2 ** self.t
        for param, grad, mom, vel in zip(self.params, grads, self.m, self.v):
            mom *= self.beta1
            mom += (1.0 - self.beta1) * grad
            vel *= self.beta2
            vel += (1.0 - self.beta2) * (grad * grad)
            param -= self.lr * (mom / b1c) / (np.sqrt(vel / b2c) + self.eps)


class HeatingNet:
    """MLP from a standardized design vector to ``log10`` quanta."""

    def __init__(self, in_dim, hidden=32, out_dim=3, stochastic=False, seed=0):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.stochastic = bool(stochastic)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.in_dim), (self.in_dim, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden), (self.hidden, self.hidden))
        self.b2 = np.zeros(self.hidden)
        self.W3 = rng.normal(0.0, np.sqrt(1.0 / self.hidden), (self.hidden, self.out_dim))
        self.b3 = np.zeros(self.out_dim)
        if self.stochastic:
            self.Ws = rng.normal(0.0, 0.01, (self.in_dim, self.hidden))
            self.bs = np.full(self.hidden, np.log(np.expm1(0.1)))
        else:
            self.Ws = None
            self.bs = None

    def param_list(self):
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        if self.stochastic:
            params.extend([self.Ws, self.bs])
        return params

    def forward(self, x, eps=None):
        """Return ``log10`` quanta.

        ``eps`` has shape ``(S, B, H)`` for a stochastic net and the return
        is ``(S, B, C)``. Without noise the return is ``(B, C)``.
        """
        x = np.asarray(x, dtype=float)
        z1 = x @ self.W1 + self.b1
        h1 = relu(z1)
        if eps is None:
            z2 = h1 @ self.W2 + self.b2
            h2 = relu(z2)
            y = h2 @ self.W3 + self.b3
            return y, (x, z1, h1, z2, h2, None, None)
        if not self.stochastic:
            raise ValueError("this network has no input-dependent noise")
        eps = np.asarray(eps, dtype=float)
        raw = x @ self.Ws + self.bs
        scale = softplus(raw)
        h1n = h1[None, :, :] + scale[None, :, :] * eps
        z2 = h1n @ self.W2 + self.b2
        h2 = relu(z2)
        y = h2 @ self.W3 + self.b3
        return y, (x, z1, h1, z2, h2, raw, scale)

    def mse_and_grad(self, x, target):
        """Mean squared error against ``target`` of shape ``(B, C)``."""
        pred, cache = self.forward(x)
        resid = pred - target
        loss = float(np.mean(resid ** 2))
        dy = (2.0 * resid) / resid.size
        grads = self._backward_deterministic(dy, cache)
        return loss, grads

    def nll(self, x, y, eps):
        pred, _cache = self.forward(x, eps)
        return _gaussian_nll(pred, y)[0]

    def nll_and_grad(self, x, y, eps):
        """Gaussian NLL of the Monte Carlo predictive distribution.

        ``y`` has shape ``(B, K, C)`` and ``eps`` has shape ``(S, B, H)``
        with ``S >= 2``. The predictive mean and variance are the mean and
        variance of the stochastic forward passes.
        """
        pred, cache = self.forward(x, eps)
        loss, dpred = _gaussian_nll(pred, y)
        grads = self._backward_stochastic(dpred, cache, eps)
        return loss, grads

    def _backward_deterministic(self, dy, cache):
        x, z1, h1, z2, h2, _raw, _scale = cache
        dW3 = h2.T @ dy
        db3 = dy.sum(axis=0)
        dh2 = dy @ self.W3.T
        dz2 = dh2 * (z2 > 0.0)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def _backward_stochastic(self, dy, cache, eps):
        x, z1, h1, z2, h2, raw, scale = cache
        dW3 = np.einsum("sbh,sbc->hc", h2, dy)
        db3 = dy.sum(axis=(0, 1))
        dh2 = dy @ self.W3.T
        dz2 = dh2 * (z2 > 0.0)
        h1n = h1[None, :, :] + scale[None, :, :] * eps
        dW2 = np.einsum("sbh,sbk->hk", h1n, dz2)
        db2 = dz2.sum(axis=(0, 1))
        dh1n = dz2 @ self.W2.T
        dh1 = dh1n.sum(axis=0)
        dscale = np.sum(dh1n * eps, axis=0)
        draw = dscale * sigmoid(raw)
        dWs = x.T @ draw
        dbs = draw.sum(axis=0)
        dz1 = dh1 * (z1 > 0.0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3, dWs, dbs]


def _gaussian_nll(samples, y):
    """NLL and ``d(loss)/d(samples)``.

    ``samples`` is ``(S, B, C)`` and ``y`` is ``(B, K, C)``. The loss is the
    mean over batch and modes of the Gaussian NLL whose mean and variance
    match the Monte Carlo samples, scored on the empirical moments of ``y``.
    """
    if samples.shape[0] < 2:
        raise ValueError("stochastic NLL needs at least two noise samples")
    mu = samples.mean(axis=0)
    diff = samples - mu
    var = np.mean(diff ** 2, axis=0) + _VAR_EPS
    y_mu = y.mean(axis=1)
    y_var = np.mean((y - y_mu[:, None, :]) ** 2, axis=1)
    resid = y_mu - mu
    nll = 0.5 * (np.log(var) + y_var / var + resid ** 2 / var)
    loss = float(nll.mean())
    scale = 1.0 / nll.size
    dmu = scale * (-resid / var)
    dvar = scale * 0.5 * (1.0 / var - y_var / var ** 2 - resid ** 2 / var ** 2)
    s_count = samples.shape[0]
    d_samples = dmu[None, :, :] / s_count + dvar[None, :, :] * (2.0 * diff / s_count)
    return loss, d_samples


class HeatingPredictor:
    """Standardized design vector in, quanta distribution out."""

    def __init__(self, net, x_mean, x_std):
        self.net = net
        self.x_mean = np.asarray(x_mean, dtype=float)
        self.x_std = np.asarray(x_std, dtype=float)

    @property
    def stochastic(self):
        return self.net.stochastic

    def standardize(self, x):
        return (np.asarray(x, dtype=float) - self.x_mean) / self.x_std

    def predict_log_samples(self, x, n_samples=32, rng=None):
        xn = self.standardize(x)
        if xn.ndim == 1:
            xn = xn.reshape(1, -1)
        if not self.stochastic:
            mean, _cache = self.net.forward(xn)
            return mean[None, :, :]
        if rng is None:
            rng = np.random.default_rng()
        eps = rng.normal(size=(int(n_samples), len(xn), self.net.hidden))
        samples, _cache = self.net.forward(xn, eps)
        return samples

    def predict_heating(self, x, n_samples=32, rng=None):
        """Predictive quanta. Totals sum the three motional modes."""
        logs = self.predict_log_samples(x, n_samples=n_samples, rng=rng)
        linear = 10.0 ** logs
        total = linear.sum(axis=-1)
        return {
            "mean": linear.mean(axis=0),
            "std": linear.std(axis=0),
            "total_mean": total.mean(axis=0),
            "total_std": total.std(axis=0),
            "log_mean": logs.mean(axis=0),
        }

    def save(self, path):
        payload = {
            "kind": np.array("stochastic" if self.stochastic else "deterministic"),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "W1": self.net.W1,
            "b1": self.net.b1,
            "W2": self.net.W2,
            "b2": self.net.b2,
            "W3": self.net.W3,
            "b3": self.net.b3,
        }
        if self.stochastic:
            payload["Ws"] = self.net.Ws
            payload["bs"] = self.net.bs
        np.savez(path, **payload)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        stochastic = str(data["kind"]) == "stochastic"
        net = HeatingNet(
            in_dim=data["W1"].shape[0],
            hidden=data["W1"].shape[1],
            out_dim=data["W3"].shape[1],
            stochastic=stochastic,
            seed=0,
        )
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            getattr(net, name)[...] = data[name]
        if stochastic:
            net.Ws[...] = data["Ws"]
            net.bs[...] = data["bs"]
        return cls(net, data["x_mean"], data["x_std"])
