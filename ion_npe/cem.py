"""Modified cross-entropy search over junction and trajectory designs.

Each iteration draws designs from a diagonal Gaussian, keeps the ones the
network predicts will heat the least, and simulates only those with the full
ion dynamics. The network is fine-tuned on the simulations, and the Gaussian
is refit to the designs with the lowest simulated heating.
"""

import numpy as np

from .dataset import Dataset, design_is_admissible
from .design import BoxGaussian
from .simulate import simulate_vectors, total_heating
from .train import finetune


def _novel(candidates, existing, space, tol=0.02):
    """Drop candidates that sit on top of a design already simulated."""
    if existing is None or len(existing) == 0:
        return np.ones(len(candidates), dtype=bool)
    unit_c = space.to_unit(candidates)
    unit_e = space.to_unit(existing)
    keep = np.ones(len(candidates), dtype=bool)
    for i, row in enumerate(unit_c):
        if np.min(np.linalg.norm(unit_e - row, axis=1)) < tol:
            keep[i] = False
    return keep


def cooptimize(
    predictor,
    dataset,
    iterations=4,
    n_samples=48,
    n_simulate=6,
    n_elite=8,
    beta=0.0,
    seed=0,
    finetune_epochs=30,
    lbfgs_maxiter=0,
    noise_samples=8,
    std_floor=0.05,
    workers=1,
    simulate_kwargs=None,
):
    """Co-optimize the junction and the trajectory.

    ``beta`` adds ``beta * predictive std`` to the screening score. Zero
    ranks designs by the predicted mean heating alone.
    """
    if len(dataset) == 0:
        raise ValueError("co-optimization needs an initial dataset")
    simulate_kwargs = dict(simulate_kwargs or {})
    simulate_kwargs.pop("seeds", None)
    space = dataset.space
    dist = BoxGaussian.around(space, std=0.15)
    elites = dataset.x[dataset.best_indices(n_elite)]
    dist.update(elites, smooth=1.0, std_floor=std_floor)
    rng = np.random.default_rng(seed)
    history = []
    for iteration in range(int(iterations)):
        candidates = dist.sample(rng, n_samples)
        pred = predictor.predict_heating(candidates, n_samples=32, rng=rng)
        score = pred["total_mean"] + float(beta) * pred["total_std"]
        order = np.argsort(score)
        chosen = []
        for index in order:
            design = space.unpack(candidates[index])
            if not design_is_admissible(design):
                continue
            if not _novel(candidates[index:index + 1], dataset.x, space)[0]:
                continue
            chosen.append(index)
            if len(chosen) == n_simulate:
                break
        if not chosen:
            dist.unit_std = np.minimum(dist.unit_std * 1.5, 0.45)
            print("  iteration {}: no new candidates, widened the search".format(iteration), flush=True)
            continue
        new_x = []
        new_q = []
        new_ideal = []
        new_length = []
        new_duration = []
        print(
            "iteration {}  screening {} designs, simulating {}".format(
                iteration, n_samples, len(chosen)
            ),
            flush=True,
        )
        results = simulate_vectors(
            candidates[chosen],
            seeds=dataset.seeds,
            workers=workers,
            **simulate_kwargs,
        )
        for index, result in zip(chosen, results):
            if not result.ok:
                print("  skip: {}".format(result.message), flush=True)
                continue
            heating = float(total_heating(result.quanta))
            print("  simulated heating {:.4g}".format(heating), flush=True)
            new_x.append(candidates[index])
            new_q.append(result.quanta)
            if np.isfinite(result.quanta_ideal).all():
                new_ideal.append(result.quanta_ideal)
            else:
                new_ideal.append(np.full_like(result.quanta, np.nan))
            new_length.append(result.length_m)
            new_duration.append(result.duration_s)
        if new_x:
            dataset = Dataset(
                np.vstack([dataset.x, np.vstack(new_x)]),
                np.vstack([dataset.quanta, np.stack(new_q)]),
                np.vstack([dataset.quanta_ideal, np.stack(new_ideal)]),
                np.concatenate([dataset.length_m, np.asarray(new_length, dtype=float)]),
                np.concatenate([dataset.duration_s, np.asarray(new_duration, dtype=float)]),
                dataset.seeds,
                space,
            )
            mask = dataset.successful()
            finetune(
                predictor,
                dataset.x[mask],
                dataset.quanta[mask],
                epochs=finetune_epochs,
                seed=seed + iteration + 1,
                noise_samples=noise_samples,
                lbfgs_maxiter=lbfgs_maxiter,
                log_every=max(finetune_epochs // 2, 1),
            )
        else:
            dist.unit_std = np.minimum(dist.unit_std * 1.5, 0.45)
        elite_idx = dataset.best_indices(n_elite)
        dist.update(dataset.x[elite_idx], smooth=0.7, std_floor=std_floor)
        best = int(elite_idx[0]) if len(elite_idx) else 0
        history.append({
            "iteration": iteration,
            "best_heating": float(dataset.total_mean()[best]),
            "n_simulated": len(dataset),
            "best_index": best,
        })
        print(
            "  best heating {:.4g} over {} simulations".format(
                history[-1]["best_heating"], history[-1]["n_simulated"]
            ),
            flush=True,
        )
    return predictor, dataset, history
