"""Unit-mean Gaussian–lognormal head, Jacobian, and a synthetic field transition."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ion_npe.surrogate.data import FieldTable, SplitTable
from ion_npe.surrogate.factorized import (
    _width_penalty,
    cloud_summary_loss,
    cold_pairwise_rank_loss,
    mean_objective,
    normalized_ratios,
    shape_objective,
    factorized_terms,
    gamma_phonon_log_prob,
    mixture_ratio_log_prob,
    phonon_log_prob,
    predictive_nll,
    sample_mixture_ratios,
    sample_truncated_unit_gaussian,
    sample_unit_gamma,
    sample_unit_lognormal,
    unit_gamma_log_prob,
    unit_gaussian_log_prob,
    unit_lognormal_log_prob,
)
from ion_npe.surrogate.layout import legal_bounds
from ion_npe.surrogate.model import ShapeNet, SurrogateConfig, SurrogateNet, SurrogatePredictor
from ion_npe.surrogate.train import train_surrogate


class FactorizedDistributionTests(unittest.TestCase):
    def test_each_component_has_unit_mean(self):
        rng = np.random.default_rng(0)
        scale = np.array([[0.15, 0.4, 1.2]])
        gaussian = sample_truncated_unit_gaussian(scale, rng, (25000, 1, 3))
        lognormal = sample_unit_lognormal(scale, rng, (25000, 1, 3))
        np.testing.assert_allclose(gaussian.mean(axis=0), 1.0, atol=0.02)
        np.testing.assert_allclose(lognormal.mean(axis=0), 1.0, atol=0.03)
        self.assertTrue(np.all(gaussian >= 0.0))
        self.assertTrue(np.all(lognormal > 0.0))

    def test_mixture_has_unit_mean_for_arbitrary_weights(self):
        rng = np.random.default_rng(1)
        scale_g = np.array([[0.2, 0.5, 1.0]])
        scale_l = np.array([[0.3, 0.8, 1.4]])
        weight = np.array([[0.0, 0.35, 1.0]])
        draw = sample_mixture_ratios(scale_g, scale_l, weight, rng, 30000)
        np.testing.assert_allclose(draw.mean(axis=0), 1.0, atol=0.025)
        self.assertTrue(np.all(draw >= 0.0))

    def test_densities_integrate_and_jacobian_is_log_mean(self):
        scale_g = torch.tensor(0.45, dtype=torch.float64)
        scale_l = torch.tensor(0.80, dtype=torch.float64)
        logit = torch.tensor(-0.2, dtype=torch.float64)
        grid = torch.linspace(1e-4, 40.0, 12000, dtype=torch.float64)
        log_p = mixture_ratio_log_prob(grid, scale_g, scale_l, logit)
        mass = torch.trapz(torch.exp(log_p), grid)
        self.assertAlmostEqual(float(mass), 1.0, delta=0.02)
        mean = torch.tensor(3.5, dtype=torch.float64)
        phonon = torch.tensor(1.7, dtype=torch.float64)
        left = phonon_log_prob(phonon, mean, scale_g, scale_l, logit)
        right = mixture_ratio_log_prob(phonon / mean, scale_g, scale_l, logit) - torch.log(mean)
        self.assertAlmostEqual(float(left), float(right), places=8)

    def test_gamma_has_unit_mean_and_broadcasts_over_seeds(self):
        rng = np.random.default_rng(2)
        shape = np.array([[0.4, 1.0, 3.0]])
        draw = sample_unit_gamma(shape, rng, (20000, 1, 3))
        np.testing.assert_allclose(draw.mean(axis=0), 1.0, atol=0.03)
        self.assertTrue(np.all(draw >= 0.0))
        grid = torch.linspace(1e-3, 30.0, 20000, dtype=torch.float64)
        mass = torch.trapz(torch.exp(unit_gamma_log_prob(grid, torch.tensor(2.0, dtype=torch.float64))), grid)
        self.assertAlmostEqual(float(mass), 1.0, delta=0.02)
        phonon = torch.rand(4, 20, 3, dtype=torch.float64) + 0.05
        mean = torch.rand(4, 3, dtype=torch.float64) + 0.2
        shapes = torch.rand(4, 3, dtype=torch.float64) + 0.3
        log_p = gamma_phonon_log_prob(phonon, mean.unsqueeze(1), shapes)
        self.assertEqual(tuple(log_p.shape), (4, 20, 3))
        self.assertTrue(torch.isfinite(log_p).all())
        ratio = unit_gamma_log_prob(phonon / mean.unsqueeze(1), shapes.unsqueeze(1))
        self.assertTrue(torch.allclose(log_p, ratio - torch.log(mean.unsqueeze(1))))

    def test_zero_phonon_is_scored_without_clamping(self):
        scale_g = torch.tensor(0.3, dtype=torch.float64)
        scale_l = torch.tensor(0.5, dtype=torch.float64)
        zero = torch.tensor(0.0, dtype=torch.float64)
        gaussian = unit_gaussian_log_prob(zero, scale_g)
        lognormal = unit_lognormal_log_prob(zero, scale_l)
        self.assertTrue(torch.isfinite(gaussian))
        self.assertEqual(float(lognormal), float("-inf"))
        mean = torch.tensor(2.0, dtype=torch.float64)
        logit = torch.tensor(0.0, dtype=torch.float64)
        at_zero = phonon_log_prob(zero, mean, scale_g, scale_l, logit)
        gaussian_only = unit_gaussian_log_prob(zero, scale_g) - math.log(2.0) - math.log(2.0)
        self.assertAlmostEqual(float(at_zero), float(gaussian_only), places=6)

    def test_ratio_loss_uses_the_observed_mean(self):
        quanta = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        observed = quanta.mean(dim=1)
        wrong = observed * 100.0
        pred = {
            "mean": wrong,
            "typical": observed.clone(),
            "mean_scale": torch.ones(1, 3, dtype=torch.float64),
            "gaussian_scale": torch.full((1, 3), 0.2, dtype=torch.float64),
            "lognormal_scale": torch.full((1, 3), 0.4, dtype=torch.float64),
            "logit": torch.full((1, 3), -1.5, dtype=torch.float64),
            "corr": None,
        }
        nll, _mse, _typical, _residual, _mean = factorized_terms(pred, quanta, mask)
        ratio = quanta / observed.unsqueeze(1)
        expected = -mixture_ratio_log_prob(
            ratio, pred["gaussian_scale"].unsqueeze(1), pred["lognormal_scale"].unsqueeze(1), pred["logit"].unsqueeze(1)
        ).sum(dim=-1).mean()
        self.assertAlmostEqual(float(nll), float(expected), places=5)
        self.assertGreater(float(_mse), 1.0)

    def test_cloud_loss_matches_each_design_and_ignores_the_predicted_mean(self):
        seeds = torch.tensor([0.97, 1.01, 0.99, 1.03, 0.98, 1.02, 1.0, 0.96], dtype=torch.float64)
        quanta = seeds.view(1, -1, 1).expand(1, -1, 3).contiguous()
        mask = torch.ones(1, seeds.numel(), dtype=torch.bool)

        def _pred(scale, mean):
            return {
                "mean": torch.full((1, 3), mean, dtype=torch.float64),
                "gaussian_scale": torch.full((1, 3), scale, dtype=torch.float64),
                "lognormal_scale": torch.full((1, 3), 0.3, dtype=torch.float64),
                "logit": torch.full((1, 3), -6.0, dtype=torch.float64),
                "corr": None,
            }

        tight = cloud_summary_loss(_pred(0.05, 1.0), quanta, mask)
        wide = cloud_summary_loss(_pred(1.2, 1.0), quanta, mask)
        shifted = cloud_summary_loss(_pred(0.05, 100.0), quanta, mask)
        self.assertLess(float(tight), float(wide))
        self.assertAlmostEqual(float(tight), float(shifted), places=5)
        gamma = {
            "mean": torch.ones(1, 3, dtype=torch.float64),
            "shape": torch.full((1, 3), 2.0, dtype=torch.float64, requires_grad=True),
        }
        cloud_summary_loss(gamma, quanta, mask).sum().backward()
        self.assertIsNotNone(gamma["shape"].grad)
        self.assertTrue(torch.isfinite(gamma["shape"].grad).all())

    def test_width_penalty_is_a_hinge_above_the_cap(self):
        def _pred(scale):
            return {
                "mean": torch.ones(1, 3, dtype=torch.float64),
                "gaussian_scale": torch.full((1, 3), scale, dtype=torch.float64),
                "lognormal_scale": torch.full((1, 3), scale, dtype=torch.float64),
                "logit": torch.zeros(1, 3, dtype=torch.float64),
            }

        self.assertEqual(float(_width_penalty(_pred(0.8))), 0.0)
        self.assertGreater(float(_width_penalty(_pred(4.0))), 0.0)

    def test_shape_is_shared_by_every_design_at_one_field(self):
        lower, upper = legal_bounds()
        config = SurrogateConfig(arch="mlp", head="gauss_lognorm", hidden=32, n_blocks=1, dropout=0.0)
        model = SurrogateNet(config)
        torch.nn.init.normal_(model.shape_head.proj.weight, std=0.2)
        design = torch.tensor(np.stack([lower, upper, lower, upper]), dtype=torch.float32)
        field = torch.tensor(
            [[0.02, 0.0004], [0.02, 0.0004], [0.69, 1.0], [0.69, 1.0]],
            dtype=torch.float32,
        )
        pred = model(design, field)
        torch.testing.assert_close(pred["gaussian_scale"][0], pred["gaussian_scale"][1])
        torch.testing.assert_close(pred["lognormal_scale"][0], pred["lognormal_scale"][1])
        torch.testing.assert_close(pred["logit"][0], pred["logit"][1])
        self.assertGreater(float((pred["gaussian_scale"][0] - pred["gaussian_scale"][2]).abs().sum()), 0.0)
        self.assertGreater(float((pred["lognormal_scale"][0] - pred["lognormal_scale"][2]).abs().sum()), 0.0)

    def test_marginal_density_matches_the_point_density_when_the_scale_vanishes(self):
        quanta = torch.tensor([[[1.2, 0.8, 1.5], [0.7, 1.1, 0.9]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        mean = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        pred = {
            "mean": mean,
            "mean_scale": torch.zeros(1, 3, dtype=torch.float64),
            "gaussian_scale": torch.full((1, 3), 0.3, dtype=torch.float64),
            "lognormal_scale": torch.full((1, 3), 0.5, dtype=torch.float64),
            "logit": torch.zeros(1, 3, dtype=torch.float64),
            "corr": None,
        }
        marginal = predictive_nll(pred, quanta, mask, n_quad=12)
        point = -phonon_log_prob(
            quanta, mean.unsqueeze(1), pred["gaussian_scale"].unsqueeze(1), pred["lognormal_scale"].unsqueeze(1), pred["logit"].unsqueeze(1)
        ).sum(dim=-1).mean()
        self.assertAlmostEqual(float(marginal), float(point), places=4)

    def test_rank_loss_prefers_the_colder_design(self):
        sigma = torch.tensor([1.0, 1.0, 1.0, 1.0])
        truth = torch.tensor([1.0, 2.0, 5.0, 9.0])
        correct = cold_pairwise_rank_loss(torch.log(truth), truth, sigma)
        reversed_score = cold_pairwise_rank_loss(torch.log(truth.flip(0)), truth, sigma)
        self.assertLess(float(correct), float(reversed_score))

    def test_zero_field_samples_match_the_mean(self):
        lower, upper = legal_bounds()
        config = SurrogateConfig(arch="mlp", head="gauss_lognorm", hidden=32, n_blocks=1, dropout=0.0)
        model = SurrogateNet(config)
        predictor = SurrogatePredictor(model, lower, upper, config.e_ref, config.log_eps, "cpu")
        design = np.stack([lower, upper, 0.5 * (lower + upper)])
        sigma = np.zeros(3)
        samples = predictor.sample_quanta(design, sigma, n_samples=6, rng=0)
        mean = predictor.arithmetic_mean(design, sigma)
        np.testing.assert_allclose(samples, np.broadcast_to(mean, samples.shape))
        self.assertTrue(np.all(samples >= 0.0))

    def test_save_and_load_preserve_the_mean(self):
        lower, upper = legal_bounds()
        config = SurrogateConfig(arch="transformer", head="gauss_lognorm", d_model=16, n_layers=1, n_heads=4, ff=32, dropout=0.0)
        model = SurrogateNet(config)
        predictor = SurrogatePredictor(model, lower, upper, config.e_ref, config.log_eps, "cpu")
        design = 0.5 * (lower + upper)
        before = predictor.arithmetic_mean(design, 1.5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            predictor.save(path)
            loaded = SurrogatePredictor.load(path, device="cpu")
        after = loaded.arithmetic_mean(design, 1.5)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_old_log_mixture_checkpoint_still_loads(self):
        path = Path("results/surrogate/mlp_gaussian.pt")
        if not path.exists():
            self.skipTest("saved log-mixture checkpoint is not in the worktree")
        loaded = SurrogatePredictor.load(path, device="cpu")
        self.assertEqual(loaded.config.head, "log_mixture")
        design = np.stack([0.5 * (loaded.lower + loaded.upper)] * 2)
        total = loaded.expected_total(design, np.array([0.0, 1.0]))
        self.assertEqual(total.shape, (2,))
        self.assertTrue(np.isfinite(total).all())

    def test_normalized_ratios_drop_a_zero_mode_and_keep_unit_mean(self):
        quanta = torch.tensor([[[2.0, 0.0, 4.0], [4.0, 0.0, 6.0]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        ratio, _observed, usable = normalized_ratios(quanta, mask)
        self.assertAlmostEqual(float(ratio[0, :, 0].mean()), 1.0, places=6)
        self.assertFalse(bool(usable[:, :, 1].any()))
        self.assertTrue(bool(usable[:, :, 2].all()))

    def test_separate_shape_depends_on_the_design(self):
        from ion_npe.surrogate.layout import JOINT_DIM

        config = SurrogateConfig(arch="mlp", head="gauss_lognorm", hidden=16, n_blocks=1, dropout=0.0)
        model = ShapeNet(config)
        torch.nn.init.normal_(model.shape_head.proj.weight, std=0.2)
        design = torch.zeros(2, JOINT_DIM)
        design[1, 0] = 1.0
        field = torch.tensor([[0.4, 0.05], [0.4, 0.05]], dtype=torch.float32)
        pred = model(design, field)
        self.assertGreater(float((pred["gaussian_scale"][0] - pred["gaussian_scale"][1]).abs().sum()), 0.0)
        quanta = torch.rand(2, 8, 3) + 0.2
        mask = torch.ones(2, 8, dtype=torch.bool)
        loss, pair = shape_objective(pred, quanta, mask, torch.tensor([0.0, 1.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.encoder.in_proj.weight.grad).all())
        self.assertEqual(float(pair[0]), 0.0)

    def test_gamma_shape_loss_backward_is_finite(self):
        from ion_npe.surrogate.layout import JOINT_DIM

        config = SurrogateConfig(arch="mlp", head="gamma", hidden=16, n_blocks=1, dropout=0.0)
        model = ShapeNet(config)
        design = torch.zeros(1, JOINT_DIM)
        field = torch.tensor([[0.4, 0.02]], dtype=torch.float32)
        quanta = torch.rand(1, 10, 3) + 0.3
        mask = torch.ones(1, 10, dtype=torch.bool)
        loss, _pair = shape_objective(model(design, field), quanta, mask, torch.tensor([1.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.shape_head.proj.weight.grad).all())

    def test_mean_loss_does_not_need_a_ratio_curve(self):
        quanta = torch.tensor([[[1.0, 2.0, 3.0], [1.2, 2.2, 3.4]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        pred = {
            "mean": torch.tensor([[1.1, 2.1, 3.2]], dtype=torch.float64, requires_grad=True),
            "typical": torch.tensor([[1.05, 2.05, 3.1]], dtype=torch.float64, requires_grad=True),
        }
        loss, _pair = mean_objective(pred, quanta, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(pred["mean"].grad).all())

    def test_network_learns_a_gaussian_to_lognormal_transition(self):
        lower, upper = legal_bounds()
        rng = np.random.default_rng(2)
        n_train, n_val, seeds = 240, 60, 10
        design = rng.uniform(lower, upper, size=(n_train + n_val, lower.size))
        sigma = np.resize(np.array([0.2, 8.0]), n_train + n_val).astype(np.float64)
        rng.shuffle(sigma)
        level = (design[:, 0] - lower[0]) / max(upper[0] - lower[0], 1e-8)
        mean = np.exp(1.5 + 0.8 * level + 0.12 * sigma)
        quanta = np.empty((design.shape[0], seeds, 3), dtype=np.float64)
        for index in range(design.shape[0]):
            if sigma[index] < 1.0:
                ratio = sample_truncated_unit_gaussian(np.full(3, 0.06), rng, (seeds, 3))
            else:
                ratio = sample_unit_lognormal(np.full(3, 0.85), rng, (seeds, 3))
            quanta[index] = mean[index] * ratio
        table = FieldTable(
            design=design,
            sigma=sigma,
            quanta=quanta,
            valid=np.ones(quanta.shape[:2], dtype=bool),
            lower=lower,
            upper=upper,
            source=np.array(["synthetic"] * len(design)),
            info={},
        )
        split = SplitTable(
            table=table,
            train=np.arange(n_train),
            val=np.arange(n_train, n_train + n_val),
            test=np.arange(n_train, n_train + n_val),
            interp=np.array([], dtype=int),
            groups=np.arange(len(design)),
            holdout_fields=np.array([]),
            seed=0,
            key_decimals=5,
            train_mean=np.zeros(lower.size),
            train_std=np.ones(lower.size),
        )
        config = SurrogateConfig(arch="mlp", head="gauss_lognorm", hidden=64, n_blocks=2, dropout=0.0, lambda_mean=1.0)
        predictor = train_surrogate(
            split, config, epochs=80, batch_size=64, patience=0, seed=0, device="cpu", log_every=0, shape_epochs=40
        )
        val = np.arange(n_train, n_train + n_val)
        predicted = predictor.arithmetic_mean(design[val], sigma[val])
        observed = quanta[val].mean(axis=1)
        log_mae = np.mean(np.abs(np.log10(predicted) - np.log10(observed)))
        self.assertLess(log_mae, 0.25)
        shape = predictor.shape_parameters(design[val], sigma[val])
        low = sigma[val] < 1.0
        high = ~low
        low_cv = np.median(np.sqrt(predictor.linear_moments(design[val][low], sigma[val][low])[1]) / predicted[low])
        high_cv = np.median(np.sqrt(predictor.linear_moments(design[val][high], sigma[val][high])[1]) / predicted[high])
        self.assertLess(low_cv, high_cv)
        self.assertLess(low_cv, 0.35)
        self.assertGreater(high_cv, 0.40)


if __name__ == "__main__":
    unittest.main()
