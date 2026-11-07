"""Active collection around the nominal path.

Candidates are drawn from a Gaussian centered on the nominal design. Inside
the low-phonon half of that pool, the designs with the largest predictive
uncertainty are simulated and added to the dataset.
"""

import numpy as np

from .dataset import design_is_admissible
from .design import BoxGaussian
from .simulate import total_heating


def _zscore(values):
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if scale < 1e-8:
        return np.zeros_like(values)
    return (values - np.mean(values)) / scale


def active_ranking(total_mean, total_std, phonon_quantile=0.5):
    """Indices to simulate, best first.

    Only the lower ``phonon_quantile`` fraction of predicted heating is
    eligible. Within that low-phonon region, a point ranks earlier when its
    predicted heating is lower and its predictive standard deviation is higher.
    """
    mean = np.asarray(total_mean, dtype=float)
    std = np.asarray(total_std, dtype=float)
    log_mean = np.log10(np.maximum(mean, 1e-30))
    log_std = np.log10(np.maximum(std, 1e-30))
    cutoff = np.quantile(log_mean, phonon_quantile)
    eligible = np.flatnonzero(log_mean <= cutoff + 1e-12)
    if len(eligible) == 0:
        eligible = np.array([int(np.argmin(log_mean))])
    score = _zscore(log_mean[eligible]) - _zscore(log_std[eligible])
    return eligible[np.argsort(score)]


def acquire(
    predictor,
    existing_x,
    space,
    rng,
    n_pool=64,
    n_take=4,
    pool_std=0.08,
    phonon_quantile=0.5,
    pool="gaussian",
    fraction=0.55,
    cap_fraction=1.0,
    seeds=(0, 1, 2, 3),
    simulate_kwargs=None,
):
    """Simulate up to ``n_take`` new designs.

    ``pool="latin"`` screens a space-filling cover of the design range.
    ``pool="gaussian"`` draws a cloud centered on the nominal design.
    """
    simulate_kwargs = dict(simulate_kwargs or {})
    if pool == "latin":
        candidates = space.sample_latin(
            rng, int(n_pool), fraction=fraction, cap_fraction=cap_fraction,
        )
    elif pool == "gaussian":
        dist = BoxGaussian.around(space, mean_x=space.nominal(), std=pool_std)
        candidates = dist.sample(rng, int(n_pool))
    else:
        raise ValueError("pool must be latin or gaussian")
    pred = predictor.predict_heating(candidates, n_samples=32, rng=rng)
    order = active_ranking(pred["total_mean"], pred["total_std"], phonon_quantile)
    chosen = []
    for index in order:
        design = space.unpack(candidates[index])
        if not design_is_admissible(design):
            continue
        if len(existing_x) and np.min(np.linalg.norm(
            space.to_unit(existing_x) - space.to_unit(candidates[index]), axis=1
        )) < 0.02:
            continue
        chosen.append(int(index))
        if len(chosen) == int(n_take):
            break
    rows = []
    quanta = []
    lengths = []
    durations = []
    predicted = []
    uncertainty = []
    if chosen:
        from .farm import get_farm

        call_kwargs = dict(simulate_kwargs)
        call_kwargs["seeds"] = [int(item) for item in seeds]
        packed = get_farm().simulate(
            "design", candidates[np.asarray(chosen, dtype=int)], call_kwargs,
        )
    else:
        packed = []
    for index, result in zip(chosen, packed):
        if not result["ok"]:
            print("  skip: {}".format(result["message"]), flush=True)
            continue
        heating = float(total_heating(result["quanta"]))
        print(
            "  acquired heating {:.4g}  predicted {:.4g} ± {:.4g}".format(
                heating, pred["total_mean"][index], pred["total_std"][index]
            ),
            flush=True,
        )
        rows.append(candidates[index])
        quanta.append(result["quanta"])
        lengths.append(result["length_m"])
        durations.append(result["duration_s"])
        predicted.append(float(pred["total_mean"][index]))
        uncertainty.append(float(pred["total_std"][index]))
    record = {
        "n_acquired": len(rows),
        "predicted_mean": float(np.mean(predicted)) if predicted else np.nan,
        "predicted_std": float(np.mean(uncertainty)) if uncertainty else np.nan,
        "simulated_mean": float(np.mean([total_heating(q) for q in quanta])) if quanta else np.nan,
    }
    if not rows:
        return None, record
    return (
        np.vstack(rows),
        np.stack(quanta),
        np.asarray(lengths, dtype=float),
        np.asarray(durations, dtype=float),
    ), record
