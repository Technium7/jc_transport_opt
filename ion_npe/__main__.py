"""Command line for simulation, training, and co-optimization.

From the project directory, with the trapsim environment:

    python -m ion_npe simulate
    python -m ion_npe collect -n 40 -o data/train.npz
    python -m ion_npe train -d data/train.npz -o data/models
    python -m ion_npe optimize -d data/train.npz -m data/models/stochastic.npz
    python -m ion_npe study -o outputs/study
    python -m ion_npe joint -o outputs/joint
    python -m ion_npe nearby -o outputs/nearby
    python -m ion_npe schedule -o outputs/schedule
    python -m ion_npe demo

On the G2 mehta partition, submit from the project directory:

    sbatch cluster/run.sbatch collect -n 400 -o data/train.npz
    sbatch --nodes=1 --ntasks=1 cluster/run.sbatch train -d data/train.npz -o data/models
    sbatch cluster/run.sbatch optimize -d data/train.npz -m data/models/stochastic.npz

Or one allocation for all three: ``sbatch cluster/pipeline.sbatch``.
``study``, ``joint``, and ``nearby`` go through ``cluster/run.sbatch`` as well.
The default request is 10 nodes at 4G each (40G). The cap is 16 nodes and 128G.
"""

import argparse
from pathlib import Path

import numpy as np

from .cem import cooptimize
from .constants import L_SCALE, NOISE_SEEDS, NOISE_STD_VM, SIM_DT, TRANSPORT_SPEED_M_S, VOLTAGE_UPDATE_S
from .dataset import Dataset, collect_dataset
from .design import DesignSpace
from .demo import run_demo
from .joint import optimize_joint
from .local import optimize_local
from .pathopt import optimize_trajectory
from .plotting import plot_trajectory
from .schedule import optimize_schedule
from .simulate import simulate_design, total_heating
from .study import run_study
from .train import train_predictor
from .trajectory import path_geometry


def _fmt(values):
    return ", ".join("{:.4g}".format(float(v)) for v in np.asarray(values).reshape(-1))


def _add_physics(parser):
    parser.add_argument("--speed", type=float, default=TRANSPORT_SPEED_M_S, help="transport speed, m/s")
    parser.add_argument("--voltage-ns", type=float, default=VOLTAGE_UPDATE_S * 1e9, help="DC update hold, nanoseconds")
    parser.add_argument("--noise", type=float, default=NOISE_STD_VM, help="stray-field amplitude std, V/m")


def _physics(args):
    return {
        "speed_m_s": args.speed,
        "voltage_update_s": args.voltage_ns * 1e-9,
        "noise_std_Vm": args.noise,
    }


def _cmd_simulate(args):
    space = DesignSpace()
    design = space.nominal_design()
    if args.z_controls is not None:
        heights = np.asarray(args.z_controls, dtype=float)
        if heights.size != len(design.interior_xyz):
            raise SystemExit(
                "--z-controls needs {} heights, one per bend control".format(
                    len(design.interior_xyz)
                )
            )
        design.interior_xyz[:, 2] = heights
    seeds = tuple(args.seeds)
    result = simulate_design(design, seeds=seeds, with_ideal=True, **_physics(args))
    if not result.ok:
        raise SystemExit(result.message)
    print("length {:.2f} um, duration {:.2f} us, hold {:.4g} ns".format(
        result.length_m * 1e6,
        result.duration_s * 1e6,
        result.n_hold * SIM_DT * 1e9,
    ))
    print("start and end height fixed at 50 um")
    print("bend entrance {:.2f} um, crossing height {:.2f} um".format(
        design.z_bend * L_SCALE * 1e6, design.z_cross * L_SCALE * 1e6
    ))
    print("path z {:.2f} to {:.2f} um".format(
        result.z_range[0] * L_SCALE * 1e6, result.z_range[1] * L_SCALE * 1e6
    ))
    print("bend controls (x, y in trap heights, z in um):")
    for index, point in enumerate(design.interior_xyz):
        print("  c{}  {:.3f}  {:.3f}  {:.2f}".format(
            index, point[0], point[1], point[2] * L_SCALE * 1e6
        ))
    print("transverse frequency {:.3f} to {:.3f} MHz".format(*result.f_trans_MHz))
    for index, seed in enumerate(seeds):
        print("seed {}  physical final quanta: {}".format(seed, _fmt(result.quanta[index])))
        print("         ideal final quanta:     {}".format(_fmt(result.quanta_ideal[index])))
        print("         peak physical quanta:   {}".format(_fmt(result.peak_quanta[index])))
        print("         noise max V/m:          {}".format(_fmt(result.noise_max_Vm[index])))
    print("mean physical total quanta: {:.6g}".format(float(total_heating(result.quanta))))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_trajectory(design, path_geometry(design), out)
        print("wrote {}".format(out))


def _cmd_collect(args):
    dataset = collect_dataset(
        args.n,
        seeds=tuple(args.seeds),
        fraction=args.fraction,
        seed=args.seed,
        **_physics(args),
    )
    dataset.save(args.out)
    print("wrote {} designs to {}".format(len(dataset), args.out))


def _cmd_train(args):
    dataset = Dataset.load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mask = dataset.successful()
    kinds = ("deterministic", "stochastic") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        print("training {}".format(kind), flush=True)
        model, _history = train_predictor(
            dataset.x[mask],
            dataset.quanta[mask],
            kind=kind,
            hidden=args.hidden,
            epochs=args.epochs,
            seed=args.seed,
            noise_samples=args.noise_samples,
            lbfgs_maxiter=args.lbfgs,
        )
        path = out / "{}.npz".format(kind)
        model.save(path)
        print("wrote {}".format(path))


def _cmd_optimize(args):
    dataset = Dataset.load(args.data)
    from .model import HeatingPredictor

    if args.model:
        model = HeatingPredictor.load(args.model)
    else:
        mask = dataset.successful()
        print("training an initial stochastic network", flush=True)
        model, _history = train_predictor(
            dataset.x[mask],
            dataset.quanta[mask],
            kind="stochastic",
            hidden=args.hidden,
            epochs=args.epochs,
            seed=args.seed,
            noise_samples=args.noise_samples,
            lbfgs_maxiter=args.lbfgs,
        )
    model, dataset, history = cooptimize(
        model,
        dataset,
        iterations=args.iterations,
        n_samples=args.samples,
        n_simulate=args.simulate,
        n_elite=args.elite,
        beta=args.beta,
        seed=args.seed,
        finetune_epochs=args.finetune_epochs,
        lbfgs_maxiter=args.lbfgs,
        noise_samples=args.noise_samples,
        simulate_kwargs=_physics(args),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset.save(out / "dataset.npz")
    model.save(out / "stochastic.npz")
    best = int(np.argmin(dataset.total_mean()))
    np.save(out / "best_design.npy", dataset.x[best])
    plot_trajectory(
        dataset.space.unpack(dataset.x[best]),
        path_geometry(dataset.space.unpack(dataset.x[best])),
        out / "best_path.png",
    )
    print("best total quanta {:.6g}".format(dataset.total_mean()[best]))
    print("wrote {}".format(out))
    for row in history:
        print("  iteration {}  best {:.6g}".format(row["iteration"], row["best_heating"]))


def _cmd_demo(args):
    run_demo(args.out, seed=args.seed)


def _cmd_nearby(args):
    optimize_local(
        args.out,
        n=args.n,
        per_round=args.per_round,
        iterations=args.iterations,
        epochs=args.epochs,
        lbfgs_maxiter=args.lbfgs,
        seed=args.seed,
        simulate_kwargs=dict(_physics(args), seeds=tuple(args.seeds)),
        cooptimize=args.cooptimize,
        warm_start=args.warm or None,
    )


def _cmd_joint(args):
    optimize_joint(
        args.out,
        n=args.n,
        iterations=args.iterations,
        active_rounds=args.active,
        n_screen=args.screen,
        n_simulate=args.simulate,
        epochs=args.epochs,
        lbfgs_maxiter=args.lbfgs,
        seed=args.seed,
        simulate_kwargs=_physics(args),
    )


def _cmd_trajectory(args):
    optimize_trajectory(
        args.out,
        n=args.n,
        iterations=args.iterations,
        epochs=args.epochs,
        lbfgs_maxiter=args.lbfgs,
        seed=args.seed,
        simulate_kwargs=_physics(args),
    )


def _cmd_schedule(args):
    optimize_schedule(
        args.out,
        n=args.n,
        iterations=args.iterations,
        epochs=args.epochs,
        lbfgs_maxiter=args.lbfgs,
        seed=args.seed,
        simulate_kwargs=_physics(args),
    )


def _cmd_study(args):
    run_study(
        args.out,
        n=args.n,
        n_fine=args.fine,
        seeds=tuple(args.seeds),
        fraction=args.fraction,
        seed=args.seed,
        epochs=args.epochs,
        lbfgs_maxiter=args.lbfgs,
        iterations=args.iterations,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ion_npe", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="integrate the nominal design, with optional heights")
    _add_physics(sim)
    sim.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sim.add_argument(
        "--z-controls", type=float, nargs="+", default=None,
        help="height of each bend control, in trap heights; default is 50 um",
    )
    sim.add_argument("--out", type=str, default="", help="optional path figure")
    sim.set_defaults(func=_cmd_simulate)

    collect = sub.add_parser("collect", help="simulate random designs into an npz dataset")
    _add_physics(collect)
    collect.add_argument("-n", type=int, default=24)
    collect.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    collect.add_argument("--fraction", type=float, default=0.5, help="box width about the nominal design, 0 to 1")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("-o", "--out", type=str, default="data/train.npz")
    collect.set_defaults(func=_cmd_collect)

    train = sub.add_parser("train", help="fit deterministic and stochastic heating networks")
    train.add_argument("-d", "--data", type=str, required=True)
    train.add_argument("-o", "--out", type=str, default="data/models")
    train.add_argument("--kind", choices=("both", "deterministic", "stochastic"), default="both")
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--lbfgs", type=int, default=40, help="L-BFGS iterations after Adam; 0 skips it")
    train.add_argument("--noise-samples", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=_cmd_train)

    opt = sub.add_parser("optimize", help="cross-entropy co-optimization")
    _add_physics(opt)
    opt.add_argument("-d", "--data", type=str, required=True)
    opt.add_argument("-m", "--model", type=str, default="", help="trained stochastic network; trained from the data if omitted")
    opt.add_argument("-o", "--out", type=str, default="outputs/optimize")
    opt.add_argument("--iterations", type=int, default=4)
    opt.add_argument("--samples", type=int, default=48)
    opt.add_argument("--simulate", type=int, default=6)
    opt.add_argument("--elite", type=int, default=8)
    opt.add_argument("--beta", type=float, default=0.0, help="weight on the predictive std in the screening score")
    opt.add_argument("--hidden", type=int, default=32)
    opt.add_argument("--epochs", type=int, default=200)
    opt.add_argument("--lbfgs", type=int, default=40, help="L-BFGS iterations after each Adam stage")
    opt.add_argument("--finetune-epochs", type=int, default=30)
    opt.add_argument("--noise-samples", type=int, default=8)
    opt.add_argument("--seed", type=int, default=0)
    opt.set_defaults(func=_cmd_optimize)

    demo = sub.add_parser("demo", help="small end-to-end run")
    demo.add_argument("-o", "--out", type=str, default="outputs/demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_cmd_demo)

    study = sub.add_parser(
        "study",
        help="center data on the nominal path, then actively collect low-phonon uncertain designs",
    )
    study.add_argument("-o", "--out", type=str, default="outputs/study")
    study.add_argument("-n", type=int, default=900, help="space-filling designs over the wider junction and trajectory range")
    study.add_argument("--fine", type=int, default=400, help="finer space-filling designs around the nominal path")
    study.add_argument("--seeds", type=int, nargs="+", default=[0])
    study.add_argument("--fraction", type=float, default=0.55, help="how far the wide sample moves from the nominal design")
    study.add_argument("--seed", type=int, default=0)
    study.add_argument("--epochs", type=int, default=60)
    study.add_argument("--lbfgs", type=int, default=25)
    study.add_argument("--iterations", type=int, default=4, help="active-learning rounds")
    study.set_defaults(func=_cmd_study)

    sched = sub.add_parser(
        "schedule",
        help="move the transport waveform gates and cross-entropy search them",
    )
    _add_physics(sched)
    sched.add_argument("-o", "--out", type=str, default="outputs/schedule")
    sched.add_argument("-n", type=int, default=180, help="waveforms drawn around the nominal time law")
    sched.add_argument("--iterations", type=int, default=8)
    sched.add_argument("--epochs", type=int, default=60)
    sched.add_argument("--lbfgs", type=int, default=25)
    sched.add_argument("--seed", type=int, default=0)
    sched.set_defaults(func=_cmd_schedule)

    traj = sub.add_parser(
        "trajectory",
        help="search arm, bend, and crossing xyz, with the ion starting and ending at 50 um",
    )
    _add_physics(traj)
    traj.add_argument("-o", "--out", type=str, default="outputs/trajectory")
    traj.add_argument("-n", type=int, default=120)
    traj.add_argument("--iterations", type=int, default=6)
    traj.add_argument("--epochs", type=int, default=40)
    traj.add_argument("--lbfgs", type=int, default=20)
    traj.add_argument("--seed", type=int, default=0)
    traj.set_defaults(func=_cmd_trajectory)

    joint = sub.add_parser(
        "joint",
        help="search the junction, the trajectory, and the waveform in one vector",
    )
    _add_physics(joint)
    joint.add_argument("-o", "--out", type=str, default="outputs/joint")
    joint.add_argument("-n", type=int, default=5000, help="space-filling training designs that vary the junction, the trajectory, and the waveform")
    joint.add_argument("--iterations", type=int, default=6, help="cross-entropy rounds; each round adds simulations")
    joint.add_argument("--active", type=int, default=4, help="active-learning rounds after the cross-entropy rounds")
    joint.add_argument("--screen", type=int, default=256, help="candidate designs scored by the network each round")
    joint.add_argument("--simulate", type=int, default=8, help="new dynamics simulations each round")
    joint.add_argument("--epochs", type=int, default=40)
    joint.add_argument("--lbfgs", type=int, default=20)
    joint.add_argument("--seed", type=int, default=0)
    joint.set_defaults(func=_cmd_joint)

    nearby = sub.add_parser(
        "nearby",
        help="start at the nominal path and iteratively collect nearby junction, trajectory, and waveform designs",
    )
    _add_physics(nearby)
    nearby.add_argument("-o", "--out", type=str, default="outputs/nearby")
    nearby.add_argument("-n", type=int, default=2000, help="designs around the nominal path before the first fit")
    nearby.add_argument("--per-round", type=int, default=1000, help="new designs collected in each proposed region")
    nearby.add_argument("--iterations", type=int, default=3, help="optimization rounds after the nominal neighborhood")
    nearby.add_argument("--epochs", type=int, default=40)
    nearby.add_argument("--lbfgs", type=int, default=15)
    nearby.add_argument("--seed", type=int, default=0)
    nearby.add_argument(
        "--seeds", type=int, nargs="+", default=list(NOISE_SEEDS),
        help="stray-field samples per design; at least {} when the field is on".format(len(NOISE_SEEDS)),
    )
    nearby.add_argument(
        "--cooptimize", action="store_true",
        help="search the junction, the trajectory, and the waveform; each may be updated or kept",
    )
    nearby.add_argument(
        "--warm", type=str, default="",
        help="npz from an earlier search; round 0 is centered on its best design",
    )
    nearby.set_defaults(func=_cmd_nearby)

    args = parser.parse_args(argv)
    from .farm import get_farm

    farm = get_farm()
    if farm.size > 1 and not farm.is_root:
        farm.serve()
        return
    try:
        args.func(args)
    except BaseException:
        if farm.size > 1 and farm.is_root:
            try:
                farm.stop()
            except Exception as err:
                print("farm shutdown: {}".format(err), flush=True)
        raise
    else:
        if farm.size > 1 and farm.is_root:
            farm.stop()


if __name__ == "__main__":
    main()
