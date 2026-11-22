"""Mean-factorized Gaussian–lognormal mixture on linear phonon number.

The network predicts the arithmetic mean of each mode and, separately, the
shape of the within-design ratio ``R = phonon / observed mean``. Both the
truncated-Gaussian component and the lognormal component are built so that
``E[R] = 1``. A third head predicts how uncertain that mean is. A predictive
draw is ``mean * exp(mean_scale * ε) * R`` with one shared standard normal
``ε`` across the three modes. ``expected_total`` stays the sum of the predicted
arithmetic means, so ranking is not the mean of that heavy-tailed draw.

The shape likelihood uses the observed design-field mean as the plug-in
normalizer. It is not asked to explain error in the predicted mean. The ratio
curve itself is a function of the field only: every design at one amplitude
shares one Gaussian scale, one lognormal scale, and one mixing weight. At
zero stray field the shape density is not evaluated and samples sit on the mean.

The joint trainer still minimizes both pieces together. ``train_mean_model``
and ``train_shape_model`` fit them separately. Neither loss reads the other
model. The separate shape model may use the design as well as the field,
because it is no longer tied to the mean head. The joint ratio curve stays a
function of the field alone. Dividing by the sample mean makes the ratios
dependent, so their product density is a working objective, not an exact
independent-sample likelihood.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2PI_HALF = 0.5 * _LOG_2PI
N_MODES = 3
# Scales below this cap are free. Above it, a hinge stops an unused
# component from walking its scale to infinity.
SCALE_CAP = 2.5
# A fraction error of 0.25 then costs about as much as a factor-of-two CV error.
FRACTION_LOSS_WEIGHT = 8.0
# A mode whose sample mean is below this floor has no usable ratio.
MEAN_FLOOR = 1e-6
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
# A CDF miss of 0.2 at one order statistic costs about 1.
QUANTILE_LOSS_WEIGHT = 25.0


def _phi_log(z):
    return -0.5 * z.square() - _LOG_2PI_HALF


def _Phi(z):
    return torch.special.ndtr(z)


def _truncation_stats(scale):
    """Mean and ``log Z`` of ``Normal(1, scale)`` truncated to ``(0, inf)``."""
    z = 1.0 / scale
    log_phi = _phi_log(z)
    log_z = torch.log(_Phi(z).clamp_min(torch.finfo(scale.dtype).tiny))
    lam = torch.exp(log_phi - log_z)
    mean_x = 1.0 + scale * lam
    alpha = -z
    var_x = scale.square() * (1.0 + alpha * lam - lam.square())
    return mean_x, log_z, var_x.clamp_min(0.0)


def unit_gaussian_log_prob(ratio, scale):
    """Log density of the unit-mean truncated Gaussian on ``R``."""
    mean_x, log_z, _var = _truncation_stats(scale)
    x = ratio * mean_x
    resid = (x - 1.0) / scale
    log_f = _phi_log(resid) - torch.log(scale) - log_z + torch.log(mean_x)
    return torch.where(ratio >= 0.0, log_f, torch.full_like(log_f, -float("inf")))


def unit_lognormal_log_prob(ratio, scale):
    """Log density of a lognormal with location ``-0.5 scale^2`` and mean 1."""
    mu = -0.5 * scale.square()
    safe = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
    log_r = torch.log(safe)
    resid = (log_r - mu) / scale
    log_f = -log_r - torch.log(scale) - _LOG_2PI_HALF - 0.5 * resid.square()
    return torch.where(ratio > 0.0, log_f, torch.full_like(log_f, -float("inf")))


def mixture_ratio_log_prob(ratio, gaussian_scale, lognormal_scale, logit):
    """Log density of the unit-mean Gaussian–lognormal mixture."""
    log_w = -F.softplus(-logit)
    log_other = -F.softplus(logit)
    gauss = log_other + unit_gaussian_log_prob(ratio, gaussian_scale)
    lognorm = log_w + unit_lognormal_log_prob(ratio, lognormal_scale)
    return torch.logsumexp(torch.stack((gauss, lognorm), dim=0), dim=0)


def phonon_log_prob(phonon, mean, gaussian_scale, lognormal_scale, logit):
    """Log density of ``mean * R``, including the ``-log(mean)`` Jacobian."""
    ratio = phonon / mean
    return mixture_ratio_log_prob(ratio, gaussian_scale, lognormal_scale, logit) - torch.log(mean)


def unit_gamma_log_prob(ratio, shape):
    """Log density of a gamma with mean 1. ``shape`` is the gamma shape."""
    safe = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
    log_f = (
        shape * torch.log(shape)
        - torch.lgamma(shape)
        + (shape - 1.0) * torch.log(safe)
        - shape * safe
    )
    return torch.where(ratio > 0.0, log_f, torch.full_like(log_f, -float("inf")))


def gamma_phonon_log_prob(phonon, mean, shape):
    """Linear-quanta gamma log density. ``shape`` may omit the seed axis."""
    ratio = phonon / mean
    while shape.ndim < ratio.ndim:
        shape = shape.unsqueeze(-2)
    return unit_gamma_log_prob(ratio, shape) - torch.log(mean)


def mixture_log_cdf(ratio, gaussian_scale, lognormal_scale, logit):
    """CDF of the unit-mean mixture. Used by the Gaussian copula."""
    mean_x, log_z, _var = _truncation_stats(gaussian_scale)
    z_mass = torch.exp(log_z)
    upper = _Phi((ratio * mean_x - 1.0) / gaussian_scale)
    lower = _Phi(-1.0 / gaussian_scale)
    gauss = ((upper - lower) / z_mass).clamp(0.0, 1.0)
    gauss = torch.where(ratio > 0.0, gauss, torch.zeros_like(gauss))
    mu = -0.5 * lognormal_scale.square()
    safe = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
    lognorm = _Phi((torch.log(safe) - mu) / lognormal_scale)
    lognorm = torch.where(ratio > 0.0, lognorm, torch.zeros_like(lognorm))
    weight = torch.sigmoid(logit)
    return (1.0 - weight) * gauss + weight * lognorm


def correlation_triplet(raw):
    """Three correlations that always form a positive-definite matrix.

    ``raw`` is ``(..., 3)`` in pair order ``(0,1), (0,2), (1,2)``.
    """
    r01 = torch.tanh(raw[..., 0])
    r02 = torch.tanh(raw[..., 1])
    bound = torch.sqrt((1.0 - r01.square()) * (1.0 - r02.square())).clamp_min(1e-6)
    r12 = r01 * r02 + bound * torch.tanh(raw[..., 2])
    return r01, r02, r12


def _correlation_matrix(r01, r02, r12):
    eye = torch.ones_like(r01)
    row0 = torch.stack((eye, r01, r02), dim=-1)
    row1 = torch.stack((r01, eye, r12), dim=-1)
    row2 = torch.stack((r02, r12, eye), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def gaussian_copula_log_prob(cdf, raw):
    """Gaussian-copula log density. ``cdf`` is ``(..., 3)`` and strictly inside (0, 1)."""
    r01, r02, r12 = correlation_triplet(raw)
    corr = _correlation_matrix(r01, r02, r12)
    latent = torch.special.ndtri(cdf.clamp(1e-6, 1.0 - 1e-6))
    batch, seeds, modes = latent.shape
    corr_flat = corr.unsqueeze(1).expand(batch, seeds, modes, modes).reshape(batch * seeds, modes, modes)
    solved = torch.linalg.solve(corr_flat, latent.reshape(batch * seeds, modes, 1))
    solved = solved.reshape(batch, seeds, modes)
    quad = (latent * (solved - latent)).sum(dim=-1)
    _sign, logdet = torch.linalg.slogdet(corr)
    return -0.5 * logdet.unsqueeze(-1) - 0.5 * quad


def _masked_mean(values, mask):
    mask_f = mask.to(dtype=values.dtype)
    counted = mask_f.sum(dim=-1).clamp_min(1.0)
    observed = (values * mask_f.unsqueeze(-1)).sum(dim=1) / counted.unsqueeze(-1)
    return observed, counted


def _masked_median(values, mask):
    """Lower median of valid seeds. ``values`` is ``(batch, seeds, modes)``."""
    filled = torch.where(mask.unsqueeze(-1), values, torch.full_like(values, float("inf")))
    ordered, _index = torch.sort(filled, dim=1)
    count = mask.to(dtype=torch.long).sum(dim=1)
    index = ((count - 1).clamp_min(0) // 2).view(-1, 1, 1).expand(-1, 1, values.shape[-1])
    median = ordered.gather(1, index).squeeze(1)
    return torch.where(count.unsqueeze(-1) > 0, median, torch.ones_like(median))


def _seed_mean_total(quanta, mask):
    total = quanta.sum(dim=-1)
    mask_f = mask.to(dtype=total.dtype)
    return (total * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)


def factorized_terms(pred, quanta, mask):
    """Within-design ratio NLL, arithmetic-mean error, and typical-value error.

    The ratio is ``phonon / observed arithmetic mean``. The predicted mean is
    not the normalizer, so mean error cannot inflate the mixture scales.
    Masked seeds are replaced before the density is evaluated. A valid zero is
    scored at zero and is not replaced by a positive floor.
    """
    mean = pred["mean"].to(dtype=torch.float64).clamp_min(1e-30)
    quanta = quanta.to(dtype=torch.float64)
    mask_f = mask.to(dtype=mean.dtype)
    safe = torch.where(mask.unsqueeze(-1), quanta, torch.ones_like(quanta))
    observed, _counted = _masked_mean(safe, mask)
    observed = observed.clamp_min(1e-30)
    ratio = safe / observed.unsqueeze(1)
    if "shape" in pred:
        log_marg = unit_gamma_log_prob(ratio, pred["shape"].to(dtype=torch.float64).unsqueeze(1))
        joint = log_marg.sum(dim=-1)
    else:
        scale_g = pred["gaussian_scale"].to(dtype=torch.float64)
        scale_l = pred["lognormal_scale"].to(dtype=torch.float64)
        logit = pred["logit"].to(dtype=torch.float64)
        log_marg = mixture_ratio_log_prob(ratio, scale_g.unsqueeze(1), scale_l.unsqueeze(1), logit.unsqueeze(1))
        joint = log_marg.sum(dim=-1)
        if pred.get("corr") is not None:
            cdf = mixture_log_cdf(ratio, scale_g.unsqueeze(1), scale_l.unsqueeze(1), logit.unsqueeze(1))
            cdf = torch.where(mask.unsqueeze(-1), cdf, torch.full_like(cdf, 0.5))
            joint = joint + gaussian_copula_log_prob(cdf, pred["corr"].to(dtype=torch.float64))
    joint = torch.where(mask, joint, torch.zeros_like(joint))
    counted = mask_f.sum(dim=-1).clamp_min(1.0)
    nll = -(joint * mask_f).sum(dim=-1) / counted
    positive_obs = observed > 0
    log_error = torch.log(mean) - torch.log(observed)
    mse = (log_error.square() * positive_obs.to(mean.dtype)).sum(dim=-1)
    mse = mse / positive_obs.to(mean.dtype).sum(dim=-1).clamp_min(1.0)
    typical = pred.get("typical", mean).to(dtype=torch.float64).clamp_min(1e-30)
    median = _masked_median(safe, mask).clamp_min(1e-30)
    typical_error = torch.log(typical) - torch.log(median)
    typical_mse = (typical_error.square() * positive_obs.to(mean.dtype)).sum(dim=-1)
    typical_mse = typical_mse / positive_obs.to(mean.dtype).sum(dim=-1).clamp_min(1.0)
    mean_scale = pred.get("mean_scale")
    if mean_scale is None:
        residual = torch.zeros_like(nll)
    else:
        delta = (torch.log(observed) - torch.log(mean)).detach()
        scale = mean_scale.to(dtype=torch.float64).clamp_min(1e-3)
        piece = 0.5 * (delta / scale).square() + torch.log(scale)
        residual = (piece * positive_obs.to(mean.dtype)).sum(dim=-1)
        residual = residual / positive_obs.to(mean.dtype).sum(dim=-1).clamp_min(1.0)
    return nll, mse, typical_mse, residual, mean


def _width_penalty(pred):
    """Hinge on scales above ``SCALE_CAP``.

    Below the cap the ratio likelihood and the per-design cloud loss set the
    width. A penalty on ``scale^2`` at every field was keeping the lognormal
    scale from reaching the width the high-field seeds need. Weighting that
    penalty by the mixture weight let an unused component run to infinity,
    because both the penalty and the likelihood gradient vanish with the weight.
    """
    if "logit" not in pred:
        reference = pred["shape"] if "shape" in pred else pred["mean"]
        return reference.new_zeros(reference.shape[0])
    gaussian = pred["gaussian_scale"].to(dtype=torch.float64)
    lognormal = pred["lognormal_scale"].to(dtype=torch.float64)
    excess = F.relu(gaussian - SCALE_CAP).square() + F.relu(lognormal - SCALE_CAP).square()
    return excess.mean(dim=-1)


def _mixture_cloud(gaussian_scale, lognormal_scale, logit):
    """Coefficient of variation and ``P(R < 1)`` of the unit-mean mixture."""
    weight = torch.sigmoid(logit)
    mean_x, _log_z, var_x = _truncation_stats(gaussian_scale)
    var_g = var_x / mean_x.square().clamp_min(1e-30)
    var_l = torch.expm1(lognormal_scale.square()).clamp_min(0.0)
    variance = (1.0 - weight) * var_g + weight * var_l
    cv = variance.clamp_min(0.0).sqrt()
    below = mixture_log_cdf(torch.ones_like(gaussian_scale), gaussian_scale, lognormal_scale, logit)
    return cv, below.clamp(0.0, 1.0)


def _gamma_cloud(shape):
    """Coefficient of variation and ``P(R < 1)`` of a unit-mean gamma.

    The regularized gamma CDF has no derivative with respect to its shape in
    this PyTorch build. For a gamma the two summaries are functions of that
    one shape, and the coefficient of variation is the one that carries the
    gradient.
    """
    safe = shape.clamp_min(1e-3)
    below = torch.special.gammainc(safe.detach(), safe.detach())
    return safe.rsqrt(), below.clamp(0.0, 1.0)


def cloud_summary_loss(pred, quanta, mask, cv_floor=1e-3):
    """Match each design's own seed cloud, not every seed's density.

    Twenty seeds do not determine a density. They do determine the coefficient
    of variation and the fraction of seeds below the design mean. Each design
    contributes one term, so one seed at many times the mean cannot outvote
    the other designs. ``-log p(R)`` remains the full-density score.
    """
    quanta = quanta.to(dtype=torch.float64)
    mask_f = mask.to(dtype=torch.float64)
    safe = torch.where(mask.unsqueeze(-1), quanta, torch.ones_like(quanta))
    observed, _counted = _masked_mean(safe, mask)
    observed = observed.clamp_min(1e-30)
    ratio = safe / observed.unsqueeze(1)
    count = mask_f.sum(dim=-1).clamp_min(1.0)
    second = ((ratio - 1.0).square() * mask_f.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
    observed_cv = second.clamp_min(0.0).sqrt()
    below = ((ratio < 1.0).to(dtype=mask_f.dtype) * mask_f.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
    if "shape" in pred:
        curve_cv, curve_below = _gamma_cloud(pred["shape"].to(dtype=torch.float64))
    else:
        curve_cv, curve_below = _mixture_cloud(
            pred["gaussian_scale"].to(dtype=torch.float64),
            pred["lognormal_scale"].to(dtype=torch.float64),
            pred["logit"].to(dtype=torch.float64),
        )
    cv_gap = (
        torch.log(curve_cv.clamp_min(cv_floor)) - torch.log(observed_cv.clamp_min(cv_floor))
    ).square()
    frac_gap = (curve_below - below).square()
    per_design = (cv_gap + FRACTION_LOSS_WEIGHT * frac_gap).mean(dim=-1)
    enough = mask_f.sum(dim=-1) >= 2.0
    return torch.where(enough, per_design, torch.zeros_like(per_design))


def cold_pairwise_rank_loss(log_score, true_total, sigma, cold_quantile=0.3):
    """Soft ranking loss that emphasizes the colder designs inside each field."""
    labels = torch.round(sigma.detach() * 1e8).to(dtype=torch.int64)
    total = log_score.new_zeros(())
    groups = 0
    for label in torch.unique(labels):
        chosen = (labels == label).nonzero(as_tuple=False).flatten()
        if chosen.numel() < 4:
            continue
        truth = true_total[chosen].detach()
        score = log_score[chosen]
        cutoff = torch.quantile(truth, float(cold_quantile))
        use = (truth[:, None] <= cutoff) & (truth[:, None] < truth[None, :])
        if not bool(use.any()):
            continue
        pair = F.softplus(score[:, None] - score[None, :])
        total = total + pair[use].mean()
        groups += 1
    if groups == 0:
        return log_score.sum() * 0.0
    return total / groups


def factorized_objective(
    pred,
    quanta,
    mask,
    sigma,
    lambda_mean,
    lambda_typical=1.0,
    lambda_scale=1.0,
    lambda_rank=0.5,
    lambda_shape=0.2,
    lambda_cloud=0.5,
):
    """Batch loss. Zero field skips the ratio density and keeps the mean losses."""
    nll, mse, typical_mse, residual, mean = factorized_terms(pred, quanta, mask)
    cloud = cloud_summary_loss(pred, quanta, mask)
    keep = mask.to(dtype=nll.dtype).sum(dim=-1) > 0
    positive = sigma.to(dtype=nll.dtype) > 0
    pair = (
        torch.where(positive, nll, torch.zeros_like(nll))
        + float(lambda_mean) * mse
        + float(lambda_typical) * typical_mse
        + float(lambda_scale) * residual
        + float(lambda_shape) * _width_penalty(pred)
        + torch.where(positive, float(lambda_cloud) * cloud, torch.zeros_like(cloud))
    )
    if not bool(keep.any()):
        raise ValueError("factorized loss received no valid samples")
    log_score = torch.log(mean.sum(dim=-1).clamp_min(1e-30))
    rank = cold_pairwise_rank_loss(log_score, _seed_mean_total(quanta, mask), sigma)
    return pair[keep].mean() + float(lambda_rank) * rank, pair


def normalized_ratios(quanta, mask, mean_floor=MEAN_FLOOR):
    """``R = n / sample mean`` on valid seeds whose mode mean clears the floor.

    The returned ratio is 1 wherever it is not usable, so a later density
    evaluation can mask those entries. The sample mean is data and is detached
    from any predicted mean.
    """
    quanta = quanta.to(dtype=torch.float64)
    mask_f = mask.to(dtype=quanta.dtype)
    safe = torch.where(mask.unsqueeze(-1), quanta, torch.zeros_like(quanta))
    count = mask_f.sum(dim=-1).clamp_min(1.0)
    observed = safe.sum(dim=1) / count.unsqueeze(-1)
    usable = mask.unsqueeze(-1) & (observed.unsqueeze(1) >= float(mean_floor))
    ratio = torch.where(usable, safe / observed.clamp_min(float(mean_floor)).unsqueeze(1), torch.ones_like(safe))
    return ratio, observed, usable


def _ratio_log_density(pred, ratio):
    if "shape" in pred:
        return unit_gamma_log_prob(ratio, pred["shape"].to(dtype=torch.float64).unsqueeze(1))
    return mixture_ratio_log_prob(
        ratio,
        pred["gaussian_scale"].to(dtype=torch.float64).unsqueeze(1),
        pred["lognormal_scale"].to(dtype=torch.float64).unsqueeze(1),
        pred["logit"].to(dtype=torch.float64).unsqueeze(1),
    )


def shape_ratio_nll(pred, quanta, mask):
    """Working per-pair ratio NLL. Pairs with no usable mode are NaN.

    Each design-field pair is one term: seeds are averaged inside the pair, so
    a pair with more realizations does not count more than a pair with fewer.
    """
    ratio, _observed, usable = normalized_ratios(quanta, mask)
    log_density = _ratio_log_density(pred, ratio)
    usable_f = usable.to(dtype=log_density.dtype)
    per_mode = -(log_density * usable_f).sum(dim=1) / usable_f.sum(dim=1).clamp_min(1.0)
    counted = usable.any(dim=1).to(dtype=per_mode.dtype)
    nll = (per_mode * counted).sum(dim=-1) / counted.sum(dim=-1).clamp_min(1.0)
    return torch.where(counted.sum(dim=-1) > 0, nll, torch.full_like(nll, float("nan"))), usable


def _ratio_cloud_loss(pred, ratio, usable, cv_floor=1e-3):
    """Match CV and ``P(R < 1)`` on the normalized seeds of each usable mode."""
    usable_f = usable.to(dtype=torch.float64)
    count = usable_f.sum(dim=1).clamp_min(1.0)
    second = ((ratio - 1.0).square() * usable_f).sum(dim=1) / count
    observed_cv = second.clamp_min(0.0).sqrt()
    below = ((ratio < 1.0).to(dtype=usable_f.dtype) * usable_f).sum(dim=1) / count
    if "shape" in pred:
        curve_cv, curve_below = _gamma_cloud(pred["shape"].to(dtype=torch.float64))
    else:
        curve_cv, curve_below = _mixture_cloud(
            pred["gaussian_scale"].to(dtype=torch.float64),
            pred["lognormal_scale"].to(dtype=torch.float64),
            pred["logit"].to(dtype=torch.float64),
        )
    cv_gap = (torch.log(curve_cv.clamp_min(cv_floor)) - torch.log(observed_cv.clamp_min(cv_floor))).square()
    frac_gap = (curve_below - below).square()
    per_mode = cv_gap + FRACTION_LOSS_WEIGHT * frac_gap
    enough = usable_f.sum(dim=1) >= 2.0
    counted = enough.to(dtype=per_mode.dtype)
    return (per_mode * counted).sum(dim=-1) / counted.sum(dim=-1).clamp_min(1.0)


def _quantile_match_loss(pred, ratio, usable):
    """Ask the mixture CDF to equal 0.1, 0.5, and 0.9 at those order statistics.

    Gamma shape has no CDF derivative in this PyTorch build, so a gamma model
    is trained through the ratio density and the coefficient of variation.
    """
    if "logit" not in pred:
        reference = pred["shape"]
        return reference.new_zeros(reference.shape[0], dtype=torch.float64)
    filled = torch.where(usable, ratio, torch.full_like(ratio, float("inf")))
    ordered, _index = torch.sort(filled, dim=1)
    count = usable.to(dtype=torch.float64).sum(dim=1)
    scale_g = pred["gaussian_scale"].to(dtype=torch.float64)
    scale_l = pred["lognormal_scale"].to(dtype=torch.float64)
    logit = pred["logit"].to(dtype=torch.float64)
    total = torch.zeros(ratio.shape[0], ratio.shape[-1], dtype=torch.float64, device=ratio.device)
    last = ordered.shape[1] - 1
    for level in QUANTILE_LEVELS:
        index = ((count - 1.0).clamp_min(0.0) * level).round().to(dtype=torch.long).clamp(0, last)
        picked = ordered.gather(1, index.unsqueeze(1)).squeeze(1)
        picked = torch.where(torch.isfinite(picked), picked, torch.ones_like(picked))
        total = total + (mixture_log_cdf(picked, scale_g, scale_l, logit) - level).square()
    per_mode = QUANTILE_LOSS_WEIGHT * total / len(QUANTILE_LEVELS)
    enough = count >= 2.0
    counted = enough.to(dtype=per_mode.dtype)
    return (per_mode * counted).sum(dim=-1) / counted.sum(dim=-1).clamp_min(1.0)


def shape_objective(
    pred,
    quanta,
    mask,
    sigma,
    lambda_shape=0.2,
    lambda_cloud=0.5,
    lambda_quantile=1.0,
):
    """Shape-only batch loss. Zero field and unusable means contribute nothing.

    The loss scores normalized seeds. It does not read a predicted mean. The
    network that produced ``pred`` may still depend on the design.
    """
    nll, usable = shape_ratio_nll(pred, quanta, mask)
    ratio, _observed, _usable = normalized_ratios(quanta, mask)
    cloud = _ratio_cloud_loss(pred, ratio, usable)
    quantile = _quantile_match_loss(pred, ratio, usable)
    finite = torch.isfinite(nll)
    nll = torch.where(finite, nll, torch.zeros_like(nll))
    positive = sigma.to(dtype=nll.dtype) > 0
    active = positive & finite & usable.any(dim=1).any(dim=-1)
    pair = (
        nll
        + float(lambda_shape) * _width_penalty(pred).to(dtype=nll.dtype)
        + float(lambda_cloud) * cloud
        + float(lambda_quantile) * quantile
    )
    pair = torch.where(active, pair, torch.zeros_like(pair))
    if not bool(active.any()):
        return nll.sum() * 0.0, pair
    return pair[active].mean(), pair


def mean_objective(pred, quanta, mask, lambda_mean=1.0, lambda_typical=1.0):
    """Log error of the arithmetic mean and of the typical (median) value.

    This loss does not read a ratio curve. Pairs with no positive sample mean
    are left out.
    """
    mean = pred["mean"].to(dtype=torch.float64).clamp_min(1e-30)
    quanta = quanta.to(dtype=torch.float64)
    safe = torch.where(mask.unsqueeze(-1), quanta, torch.ones_like(quanta))
    observed, _counted = _masked_mean(safe, mask)
    positive = observed > 0
    log_error = torch.log(mean) - torch.log(observed.clamp_min(1e-30))
    mse = (log_error.square() * positive.to(mean.dtype)).sum(dim=-1)
    mse = mse / positive.to(mean.dtype).sum(dim=-1).clamp_min(1.0)
    typical = pred.get("typical", mean).to(dtype=torch.float64).clamp_min(1e-30)
    median = _masked_median(safe, mask).clamp_min(1e-30)
    typical_error = torch.log(typical) - torch.log(median)
    typical_mse = (typical_error.square() * positive.to(mean.dtype)).sum(dim=-1)
    typical_mse = typical_mse / positive.to(mean.dtype).sum(dim=-1).clamp_min(1.0)
    pair = float(lambda_mean) * mse + float(lambda_typical) * typical_mse
    keep = mask.to(dtype=pair.dtype).sum(dim=-1) > 0
    keep = keep & positive.any(dim=-1)
    if not bool(keep.any()):
        raise ValueError("mean loss received no positive sample means")
    return pair[keep].mean(), pair


def predictive_nll(pred, quanta, mask, n_quad=16):
    """Marginal NLL of raw phonons, integrating the shared mean shock."""
    if quanta.shape[0] > 256 and pred.get("corr") is not None:
        parts = []
        for start in range(0, quanta.shape[0], 256):
            stop = start + 256
            chunk = {
                key: value[start:stop] if torch.is_tensor(value) else value
                for key, value in pred.items()
            }
            parts.append(_predictive_nll_chunk(chunk, quanta[start:stop], mask[start:stop], n_quad))
        return torch.cat(parts, dim=0)
    return _predictive_nll_chunk(pred, quanta, mask, n_quad)


def _predictive_nll_chunk(pred, quanta, mask, n_quad):
    mean = pred["mean"].to(dtype=torch.float64).clamp_min(1e-30)
    mean_scale = pred.get("mean_scale")
    if mean_scale is None:
        mean_scale = torch.zeros_like(mean)
    else:
        mean_scale = mean_scale.to(dtype=torch.float64)
    quanta = quanta.to(dtype=torch.float64)
    safe = torch.where(mask.unsqueeze(-1), quanta, torch.ones_like(quanta))
    eps, weight = _standard_nodes(n_quad, mean.dtype, mean.device)
    level = mean.unsqueeze(0) * torch.exp(mean_scale.unsqueeze(0) * eps[:, None, None])
    level = level.unsqueeze(2)
    phon = safe.unsqueeze(0)
    if "shape" in pred:
        shape = pred["shape"].to(dtype=torch.float64).view(1, -1, 1, mean.shape[-1])
        joint = gamma_phonon_log_prob(phon, level, shape).sum(dim=-1)
    else:
        scale_g = pred["gaussian_scale"].to(dtype=torch.float64).view(1, -1, 1, mean.shape[-1])
        scale_l = pred["lognormal_scale"].to(dtype=torch.float64).view(1, -1, 1, mean.shape[-1])
        logit = pred["logit"].to(dtype=torch.float64).view(1, -1, 1, mean.shape[-1])
        joint = phonon_log_prob(phon, level, scale_g, scale_l, logit).sum(dim=-1)
        if pred.get("corr") is not None:
            cdf = mixture_log_cdf(phon / level, scale_g, scale_l, logit)
            cdf = torch.where(mask.reshape(1, mask.shape[0], mask.shape[1], 1), cdf, torch.full_like(cdf, 0.5))
            quad, batch, seeds, _modes = cdf.shape
            flat = cdf.reshape(quad * batch, seeds, _modes)
            raw = pred["corr"].to(dtype=torch.float64).unsqueeze(0).expand(quad, batch, -1).reshape(quad * batch, -1)
            joint = joint + gaussian_copula_log_prob(flat, raw).reshape(quad, batch, seeds)
    joint = torch.where(mask.unsqueeze(0), joint, torch.zeros_like(joint))
    log_marg = torch.logsumexp(joint + torch.log(weight.clamp_min(1e-30)).view(-1, 1, 1), dim=0)
    mask_f = mask.to(dtype=mean.dtype)
    counted = mask_f.sum(dim=-1).clamp_min(1.0)
    return -(log_marg * mask_f).sum(dim=-1) / counted


def _standard_nodes(n, dtype, device):
    from numpy.polynomial.hermite import hermgauss

    from .torch_compat import to_tensor

    nodes, weights = hermgauss(int(n))
    eps = np.sqrt(2.0) * nodes
    prob = weights / np.sqrt(np.pi)
    return (
        to_tensor(eps, dtype=dtype, device=device),
        to_tensor(prob, dtype=dtype, device=device),
    )


class ArithmeticMeanHead(nn.Module):
    """Arithmetic mean in quanta. Zero field uses only the baseline."""

    def __init__(self, in_dim, n_modes=N_MODES):
        super().__init__()
        self.base = nn.Linear(int(in_dim), int(n_modes))
        self.delta = nn.Linear(int(in_dim), int(n_modes))
        self.tail = nn.Linear(int(in_dim), int(n_modes))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        nn.init.zeros_(self.base.bias)

    def set_log_bias(self, log_mean):
        from .torch_compat import to_tensor

        bias = to_tensor(np.asarray(log_mean, dtype=np.float64), dtype=self.base.bias.dtype, device=self.base.bias.device)
        if tuple(bias.shape) != tuple(self.base.bias.shape):
            raise ValueError("log mean bias must match the number of modes")
        with torch.no_grad():
            self.base.bias.copy_(bias)

    def forward(self, hidden, field):
        strength = field.square().sum(dim=-1, keepdim=True).sqrt()
        log_typical = self.base(hidden) + strength * self.delta(hidden)
        log_mean = log_typical + strength * self.tail(hidden)
        return log_mean.exp().clamp_min(1e-30), log_typical.exp().clamp_min(1e-30)


def field_shape_features(field):
    """Field coordinates for the shared ratio curve.

    The stored field vector is ``[log(1+σ/E), (σ/E)^2]``. ``log(σ/E + 0.01)``
    keeps 0.1 V/m and 10 V/m on the same scale. No design coordinate enters.
    """
    ratio = torch.expm1(field[..., :1].clamp(max=20.0))
    return torch.cat((field, torch.log(ratio + 0.01)), dim=-1)


class ShapeHead(nn.Module):
    """Gaussian scale, lognormal scale, and lognormal logit, shared by field."""

    def __init__(self, n_modes=N_MODES, scale_floor=1e-3, copula=False, width=32):
        super().__init__()
        self.n_modes = int(n_modes)
        self.scale_floor = float(scale_floor)
        self.copula = bool(copula)
        self.trunk = nn.Sequential(nn.Linear(3, int(width)), nn.Tanh())
        self.proj = nn.Linear(int(width), self.n_modes * 3)
        self.skip = nn.Linear(3, self.n_modes * 3)
        self.corr = nn.Linear(int(width), 3) if self.copula else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trunk[0].weight)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data.mul_(0.1)
        nn.init.xavier_uniform_(self.skip.weight)
        self.skip.weight.data.mul_(0.1)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.skip.bias)
        bias = self.proj.bias.data
        gauss_bias = _inv_softplus(0.05)
        lognorm_bias = _inv_softplus(0.40)
        for mode in range(self.n_modes):
            bias[mode * 3] = gauss_bias
            bias[mode * 3 + 1] = lognorm_bias
            bias[mode * 3 + 2] = -1.5
        if self.corr is not None:
            nn.init.zeros_(self.corr.weight)
            nn.init.zeros_(self.corr.bias)

    def forward(self, field):
        features = field_shape_features(field)
        hidden = self.trunk(features)
        raw = (self.proj(hidden) + self.skip(features)).view(-1, self.n_modes, 3)
        gaussian_scale = F.softplus(raw[..., 0]) + self.scale_floor
        lognormal_scale = F.softplus(raw[..., 1]) + self.scale_floor
        logit = raw[..., 2]
        corr = None if self.corr is None else self.corr(hidden)
        return gaussian_scale, lognormal_scale, logit, corr


class ConditionalShapeHead(nn.Module):
    """Mixture parameters from an encoder state that has seen the design."""

    def __init__(self, in_dim, n_modes=N_MODES, scale_floor=1e-3, width=32):
        super().__init__()
        self.n_modes = int(n_modes)
        self.scale_floor = float(scale_floor)
        self.trunk = nn.Sequential(nn.Linear(int(in_dim), int(width)), nn.Tanh())
        self.proj = nn.Linear(int(width), self.n_modes * 3)
        self.skip = nn.Linear(int(in_dim), self.n_modes * 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trunk[0].weight)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data.mul_(0.1)
        nn.init.xavier_uniform_(self.skip.weight)
        self.skip.weight.data.mul_(0.1)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.skip.bias)
        bias = self.proj.bias.data
        gauss_bias = _inv_softplus(0.05)
        lognorm_bias = _inv_softplus(0.40)
        for mode in range(self.n_modes):
            bias[mode * 3] = gauss_bias
            bias[mode * 3 + 1] = lognorm_bias
            bias[mode * 3 + 2] = -1.5

    def forward(self, hidden):
        raw = (self.proj(self.trunk(hidden)) + self.skip(hidden)).view(-1, self.n_modes, 3)
        gaussian_scale = F.softplus(raw[..., 0]) + self.scale_floor
        lognormal_scale = F.softplus(raw[..., 1]) + self.scale_floor
        logit = raw[..., 2]
        return gaussian_scale, lognormal_scale, logit


class ConditionalGammaHead(nn.Module):
    """Gamma shape from an encoder state that has seen the design."""

    def __init__(self, in_dim, n_modes=N_MODES, scale_floor=1e-3, width=32):
        super().__init__()
        self.floor = float(scale_floor)
        self.trunk = nn.Sequential(nn.Linear(int(in_dim), int(width)), nn.Tanh())
        self.proj = nn.Linear(int(width), int(n_modes))
        self.skip = nn.Linear(int(in_dim), int(n_modes))
        nn.init.xavier_uniform_(self.trunk[0].weight)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data.mul_(0.1)
        nn.init.xavier_uniform_(self.skip.weight)
        self.skip.weight.data.mul_(0.1)
        nn.init.constant_(self.proj.bias, _inv_softplus(5.0))
        nn.init.zeros_(self.skip.bias)

    def forward(self, hidden):
        raw = F.softplus(self.proj(self.trunk(hidden)) + self.skip(hidden)) + self.floor
        return raw.clamp(max=1.0e4)


class GammaShapeHead(nn.Module):
    """Per-mode gamma shape, shared by every design at the same field."""

    def __init__(self, n_modes=N_MODES, scale_floor=1e-3, width=32):
        super().__init__()
        self.floor = float(scale_floor)
        self.trunk = nn.Sequential(nn.Linear(3, int(width)), nn.Tanh())
        self.proj = nn.Linear(int(width), int(n_modes))
        self.skip = nn.Linear(3, int(n_modes))
        nn.init.xavier_uniform_(self.trunk[0].weight)
        nn.init.zeros_(self.trunk[0].bias)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.weight.data.mul_(0.1)
        nn.init.xavier_uniform_(self.skip.weight)
        self.skip.weight.data.mul_(0.1)
        nn.init.constant_(self.proj.bias, _inv_softplus(5.0))
        nn.init.zeros_(self.skip.bias)

    def forward(self, field):
        features = field_shape_features(field)
        raw = F.softplus(self.proj(self.trunk(features)) + self.skip(features)) + self.floor
        return raw.clamp(max=1.0e4)


class MeanUncertaintyHead(nn.Module):
    """Log-scale of a shared shock on the predicted arithmetic mean."""

    def __init__(self, in_dim, n_modes=N_MODES, floor=0.05):
        super().__init__()
        self.floor = float(floor)
        self.proj = nn.Linear(int(in_dim), int(n_modes))
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, _inv_softplus(max(1.0 - self.floor, 1e-3)))

    def forward(self, hidden, field):
        return F.softplus(self.proj(torch.cat((hidden, field), dim=-1))) + self.floor


def _inv_softplus(value):
    value = float(value)
    return math.log(math.expm1(value))


def ratio_variance(gaussian_scale, lognormal_scale, weight):
    """Variance of ``R`` for a Gaussian–lognormal mixture with mean 1."""
    gaussian_scale = np.asarray(gaussian_scale, dtype=np.float64)
    lognormal_scale = np.asarray(lognormal_scale, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    from scipy.special import erf

    z = 1.0 / gaussian_scale
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    mass = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    lam = phi / np.maximum(mass, 1e-30)
    mean_x = 1.0 + gaussian_scale * lam
    alpha = -z
    var_x = gaussian_scale ** 2 * (1.0 + alpha * lam - lam ** 2)
    var_g = np.maximum(var_x, 0.0) / mean_x ** 2
    var_l = np.expm1(lognormal_scale ** 2)
    return (1.0 - weight) * var_g + weight * np.maximum(var_l, 0.0)


def gamma_ratio_variance(shape):
    shape = np.asarray(shape, dtype=np.float64)
    return 1.0 / np.maximum(shape, 1e-30)


def sample_truncated_unit_gaussian(scale, rng, size):
    """Draw the unit-mean truncated Gaussian."""
    scale = np.asarray(scale, dtype=np.float64)
    from scipy.special import ndtr, ndtri

    z = 1.0 / scale
    mass = ndtr(z)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    mean_x = 1.0 + scale * (phi / np.maximum(mass, 1e-30))
    u = rng.random(size)
    u = (1.0 - mass) + u * mass
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    x = 1.0 + scale * ndtri(u)
    return np.maximum(x / mean_x, 0.0)


def sample_unit_lognormal(scale, rng, size):
    scale = np.asarray(scale, dtype=np.float64)
    mu = -0.5 * scale ** 2
    return np.exp(mu + scale * rng.normal(size=size))


def sample_unit_gamma(shape, rng, size):
    shape = np.asarray(shape, dtype=np.float64)
    # numpy Gamma uses scale; mean = shape * scale = 1.
    return rng.gamma(shape, 1.0 / np.maximum(shape, 1e-30), size=size)


def sample_mixture_ratios(gaussian_scale, lognormal_scale, weight, rng, n_samples, corr=None):
    """Draw ``(n_samples, batch, modes)`` unit-mean ratios."""
    gaussian_scale = np.asarray(gaussian_scale, dtype=np.float64)
    lognormal_scale = np.asarray(lognormal_scale, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    batch, modes = weight.shape
    if corr is None:
        take_lognorm = rng.random((int(n_samples), batch, modes)) < weight[None, :, :]
        gauss = sample_truncated_unit_gaussian(gaussian_scale, rng, (int(n_samples), batch, modes))
        lognorm = sample_unit_lognormal(lognormal_scale, rng, (int(n_samples), batch, modes))
        return np.where(take_lognorm, lognorm, gauss)
    return _sample_copula_ratios(gaussian_scale, lognormal_scale, weight, corr, rng, int(n_samples))


def _sample_copula_ratios(gaussian_scale, lognormal_scale, weight, corr_raw, rng, n_samples):
    from scipy.special import ndtr

    batch = weight.shape[0]
    raw = np.asarray(corr_raw, dtype=np.float64)
    r01 = np.tanh(raw[:, 0])
    r02 = np.tanh(raw[:, 1])
    bound = np.sqrt(np.maximum((1.0 - r01 ** 2) * (1.0 - r02 ** 2), 1e-12))
    r12 = r01 * r02 + bound * np.tanh(raw[:, 2])
    eye = np.ones(batch)
    corr = np.stack((
        np.stack((eye, r01, r02), axis=-1),
        np.stack((r01, eye, r12), axis=-1),
        np.stack((r02, r12, eye), axis=-1),
    ), axis=-2)
    chol = np.linalg.cholesky(corr)
    latent = rng.normal(size=(n_samples, batch, N_MODES))
    z = np.einsum("bij,nbj->nbi", chol, latent)
    uniforms = ndtr(z)
    out = np.empty_like(z)
    for mode in range(N_MODES):
        out[:, :, mode] = _invert_mixture_cdf(
            uniforms[:, :, mode],
            gaussian_scale[:, mode],
            lognormal_scale[:, mode],
            weight[:, mode],
        )
    return out


def _invert_mixture_cdf(uniform, gaussian_scale, lognormal_scale, weight):
    """Invert one marginal. ``uniform`` is ``(n_samples, batch)``."""
    low = np.full_like(uniform, -18.0)
    high = np.full_like(uniform, 8.0)
    for _ in range(40):
        mid = 0.5 * (low + high)
        cdf = _mixture_cdf_numpy(np.exp(mid), gaussian_scale, lognormal_scale, weight)
        low = np.where(cdf < uniform, mid, low)
        high = np.where(cdf < uniform, high, mid)
    return np.exp(0.5 * (low + high))


def _mixture_cdf_numpy(ratio, gaussian_scale, lognormal_scale, weight):
    from scipy.special import ndtr

    z = 1.0 / gaussian_scale
    mass = ndtr(z)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    mean_x = 1.0 + gaussian_scale * (phi / np.maximum(mass, 1e-30))
    gauss = (ndtr((ratio * mean_x - 1.0) / gaussian_scale) - ndtr(-z)) / np.maximum(mass, 1e-30)
    gauss = np.clip(gauss, 0.0, 1.0)
    mu = -0.5 * lognormal_scale ** 2
    lognorm = ndtr((np.log(np.maximum(ratio, 1e-30)) - mu) / lognormal_scale)
    return (1.0 - weight) * gauss + weight * lognorm
