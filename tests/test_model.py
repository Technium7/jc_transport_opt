"""Network gradients and a short training run."""

import unittest

import numpy as np

from ion_npe.model import HeatingNet
from ion_npe.train import train_predictor


def _gradient_error(loss_and_grad, loss_only, params_and_grads):
    step = 1e-6
    worst = 0.0
    for param, grad in params_and_grads:
        flat_g = np.asarray(grad).reshape(-1)
        flat_p = np.asarray(param).reshape(-1)
        for index in range(flat_p.size):
            original = float(flat_p[index])
            flat_p[index] = original + step
            plus = loss_only()
            flat_p[index] = original - step
            minus = loss_only()
            flat_p[index] = original
            numerical = (plus - minus) / (2.0 * step)
            scale = max(1.0, abs(numerical), abs(flat_g[index]))
            worst = max(worst, abs(numerical - flat_g[index]) / scale)
    return worst


class ModelTests(unittest.TestCase):
    def test_stochastic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        net = HeatingNet(3, hidden=4, out_dim=2, stochastic=True, seed=1)
        x = rng.normal(size=(3, 3))
        y = rng.normal(size=(3, 2, 2))
        eps = rng.normal(size=(4, 3, 4))
        loss, grads = net.nll_and_grad(x, y, eps)
        self.assertAlmostEqual(loss, net.nll(x, y, eps), places=10)
        worst = _gradient_error(
            net.nll_and_grad,
            lambda: net.nll(x, y, eps),
            list(zip(net.param_list(), grads)),
        )
        self.assertLess(worst, 1e-4)

    def test_deterministic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        net = HeatingNet(3, hidden=4, out_dim=2, stochastic=False, seed=2)
        x = rng.normal(size=(5, 3))
        target = rng.normal(size=(5, 2))
        loss, grads = net.mse_and_grad(x, target)
        worst = _gradient_error(
            net.mse_and_grad,
            lambda: net.mse_and_grad(x, target)[0],
            list(zip(net.param_list(), grads)),
        )
        self.assertGreater(loss, 0.0)
        self.assertLess(worst, 1e-4)

    def test_training_loss_falls(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(48, 4))
        signal = np.sin(x[:, 0]) + 0.3 * x[:, 1]
        log_q = np.column_stack([signal, 0.5 * signal, -0.2 * signal])
        quanta = 10.0 ** log_q
        quanta = np.repeat(quanta[:, None, :], 3, axis=1)
        quanta = quanta * np.exp(0.05 * rng.normal(size=quanta.shape))
        model, history = train_predictor(
            x, quanta, kind="stochastic", hidden=16, epochs=40,
            seed=0, log_every=0, noise_samples=4, val_fraction=0.0,
        )
        self.assertLess(history["train"][-1], 0.7 * history["train"][0])
        pred = model.predict_heating(x[:6], n_samples=8, rng=rng)
        self.assertEqual(pred["mean"].shape, (6, 3))
        self.assertTrue(np.all(pred["mean"] > 0.0))

    def test_lbfgs_lowers_the_loss_after_a_short_adam_run(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(36, 4))
        signal = np.sin(x[:, 0]) + 0.3 * x[:, 1]
        log_q = np.column_stack([signal, 0.5 * signal, -0.2 * signal])
        quanta = 10.0 ** log_q
        quanta = np.repeat(quanta[:, None, :], 3, axis=1)
        quanta = quanta * np.exp(0.05 * rng.normal(size=quanta.shape))
        _model, history = train_predictor(
            x, quanta, kind="stochastic", hidden=16, epochs=4,
            seed=0, log_every=0, noise_samples=4, val_fraction=0.0,
            lbfgs_maxiter=12,
        )
        self.assertGreater(len(history["lbfgs"]), 1)
        self.assertLess(min(history["lbfgs"]), history["lbfgs"][0])


if __name__ == "__main__":
    unittest.main()
