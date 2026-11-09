"""Figures for a trajectory, training, and the cross-entropy search."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .constants import ION_HEIGHT_M, L_SCALE, TRANSPORT_SPEED_M_S
from .simulate import total_heating
from .trajectory import (
    arm_polygon,
    control_polygon,
    dynamics_grid,
    one_sided_time_gates,
    path_geometry,
)

# Same electrode system as ``control._responses`` / ``build_junction``.
_RF_FACE = "#4C78A8"
_DC_FACE = "#F2C14E"
_RF_EDGE = "#2F5A80"
_DC_EDGE = "#A67C1F"


def _draw_junction_electrodes(ax, junction, rf_alpha=0.38, dc_alpha=0.28, lw=0.25):
    """Fill the RF and DC polygons used for the unit-volt response function."""
    for electrode in junction:
        if electrode.name == "r":
            face, edge, alpha, zorder = _RF_FACE, _RF_EDGE, rf_alpha, 1
        else:
            face, edge, alpha, zorder = _DC_FACE, _DC_EDGE, dc_alpha, 0
        for poly in electrode.paths:
            closed = np.vstack([poly, poly[:1]])
            ax.fill(
                closed[:, 0], closed[:, 1],
                facecolor=face, edgecolor=edge, alpha=alpha, lw=lw, zorder=zorder,
            )


def _electrode_xy(junction, dc_only=False):
    """Stack electrode vertices from the response-function junction.

    The RF rails continue to the chip edge. Framing plots on that full span
    hides the segmented DC electrodes, so ``dc_only`` keeps the DC extents.
    """
    points = []
    for electrode in junction:
        if dc_only and electrode.name == "r":
            continue
        for poly in electrode.paths:
            points.append(np.asarray(poly, dtype=float)[:, :2])
    if not points:
        return np.zeros((0, 2))
    return np.vstack(points)


def plot_trajectory(design, path, out):
    """RF and DC electrodes, the path, and the bend control points.

    The electrode fill is the same ``build_junction`` system used to compute
    the DC/RF response function. Squares are the clamped ends of the bend.
    Triangles are the fixed arm controls and circles are the bend controls.
    Bend controls are RF-null radius and height. The ion still starts and
    ends at 50 um.
    """
    from .junction import build_junction

    controls = control_polygon(design)
    free = controls[2:-2]
    fixed = np.vstack([controls[:2], controls[-2:]])
    arms = arm_polygon(design)[2:-2]
    if path is None:
        path = path_geometry(design)
    height_um = path[:, 2] * L_SCALE * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    ax = axes[0]
    junction = build_junction(design.junction)
    _draw_junction_electrodes(ax, junction)
    z_lo = float(height_um.min())
    z_hi = float(height_um.max())
    if z_hi - z_lo < 0.5:
        z_lo, z_hi = 45.0, 55.0
    color = ax.scatter(
        path[:, 0], path[:, 1], c=height_um, s=8, cmap="viridis",
        vmin=z_lo, vmax=z_hi, zorder=3,
    )
    fig.colorbar(color, ax=ax, fraction=0.046, pad=0.04, label="path height (μm)")
    ax.plot(controls[:, 0], controls[:, 1], color="#E45756", lw=0.9, ls="--", zorder=4)
    ax.scatter(
        fixed[:, 0], fixed[:, 1], s=28, marker="s", color="#333333",
        zorder=5, label="bend ends",
    )
    ax.scatter(
        arms[:, 0], arms[:, 1], s=32, marker="^", facecolors="none",
        edgecolors="#54A24B", linewidths=1.2, zorder=6, label="arm controls",
    )
    ax.scatter(
        free[:, 0], free[:, 1], s=32, marker="o", facecolors="none",
        edgecolors="#E45756", linewidths=1.2, zorder=6, label="bend controls",
    )
    span = np.vstack([path[:, :2], controls[:, :2], _electrode_xy(junction, dc_only=True)])
    pad = 0.35
    ax.set_xlim(span[:, 0].min() - pad, span[:, 0].max() + pad)
    ax.set_ylim(span[:, 1].min() - pad, span[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (trap heights)")
    ax.set_ylabel("y (trap heights)")
    ax.set_title("trajectory and control points")
    ax.legend(
        handles=[
            Patch(facecolor=_RF_FACE, edgecolor=_RF_EDGE, alpha=0.5, label="RF"),
            Patch(facecolor=_DC_FACE, edgecolor=_DC_EDGE, alpha=0.5, label="DC"),
            Line2D([0], [0], marker="s", color="#333333", ls="none", label="bend ends"),
            Line2D(
                [0], [0], marker="^", color="#54A24B", ls="none",
                markerfacecolor="none", label="arm controls",
            ),
            Line2D(
                [0], [0], marker="o", color="#E45756", ls="none",
                markerfacecolor="none", label="bend controls",
            ),
        ],
        frameon=False, fontsize=8, loc="upper left",
    )
    inset = ax.inset_axes([0.20, 0.55, 0.74, 0.40])
    _draw_junction_electrodes(inset, junction, rf_alpha=0.45, dc_alpha=0.32)
    inset.scatter(
        path[:, 0], path[:, 1], c=height_um, s=8, cmap="viridis",
        vmin=z_lo, vmax=z_hi, zorder=3,
    )
    inset.plot(controls[:, 0], controls[:, 1], color="#E45756", lw=0.9, ls="--", zorder=4)
    inset.scatter(fixed[:, 0], fixed[:, 1], s=22, marker="s", color="#333333", zorder=5)
    inset.scatter(
        free[:, 0], free[:, 1], s=28, marker="o", facecolors="none",
        edgecolors="#E45756", linewidths=1.1, zorder=6,
    )
    for index, point in enumerate(free):
        inset.annotate(
            "c{}".format(index),
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=6.5,
            color="#E45756",
        )
    inset.set_xlim(controls[:, 0].min() - 0.08, controls[:, 0].max() + 0.08)
    inset.set_ylim(controls[:, 1].min() - 0.04, max(controls[:, 1].max() + 0.06, 0.16))
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("bend controls", fontsize=8)
    for spine in inset.spines.values():
        spine.set_color("#E45756")
    ax.indicate_inset_zoom(inset, edgecolor="#E45756")

    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(step)])
    axes[1].plot(distance, height_um, color="#4C78A8", label="path")
    axes[1].axhline(ION_HEIGHT_M * 1e6, color="#333333", lw=0.7, ls=":", label="50 μm")
    for index, point in enumerate(controls):
        nearest = int(np.argmin((path[:, 0] - point[0]) ** 2 + (path[:, 1] - point[1]) ** 2))
        marker = "o" if 2 <= index < len(controls) - 2 else "s"
        color = "#E45756" if marker == "o" else "#333333"
        axes[1].scatter(
            distance[nearest], point[2] * L_SCALE * 1e6,
            s=36, marker=marker, color=color, zorder=3,
        )
    axes[1].set_xlabel("distance along the path (trap heights)")
    axes[1].set_ylabel("z (μm)")
    axes[1].set_title("height and control points")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _motion(design, n=480):
    """Path samples of one waveform, with distance in metres and speed in m/s."""
    preview = path_geometry(design, n=80)
    length_m = float(np.sum(np.linalg.norm(np.diff(preview, axis=0), axis=1))) * L_SCALE
    grid = dynamics_grid(
        design,
        TRANSPORT_SPEED_M_S,
        L_SCALE,
        max(length_m / TRANSPORT_SPEED_M_S / float(n), 1e-12),
    )
    xyz = grid["xyz"]
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * L_SCALE
    distance = np.concatenate([[0.0], np.cumsum(step)])
    time = np.linspace(0.0, grid["duration_s"], len(xyz))
    speed = np.gradient(distance, time)
    return xyz, distance, speed


def _colored_path(ax, xy, speed, vmin, vmax, lw=2.0):
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    drawn = LineCollection(segments, cmap="viridis", norm=plt.Normalize(vmin, vmax), linewidths=lw, zorder=3)
    drawn.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection(drawn)
    return drawn


def _height_profile(design):
    """Arc length in μm and ion height in μm. Height does not follow the speed waveform."""
    path = path_geometry(design, n=1201)
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance_um = np.concatenate([[0.0], np.cumsum(step)]) * L_SCALE * 1e6
    return distance_um, path[:, 2] * L_SCALE * 1e6


def plot_best_design(design, out, reference=None, total_quanta=None, title=None):
    """Junction, bend controls, transport waveform, and height along the path.

    The path is colored by the ion's speed. The filled electrodes are the RF
    rails and segmented DC electrodes from ``build_junction``, the same set
    used to compute the unit-volt response function. ``reference`` is the
    nominal design, drawn dashed on the junction, the bend, the speed, and
    the height.
    """
    from .junction import build_junction

    xyz, distance, speed = _motion(design)
    controls = control_polygon(design)
    free = controls[2:-2]
    fixed = np.vstack([controls[:2], controls[-2:]])
    vmax = float(np.max(speed))
    if reference is not None:
        ref_xyz, ref_distance, ref_speed = _motion(reference)
        vmax = max(vmax, float(np.max(ref_speed)))
    else:
        ref_xyz = ref_distance = ref_speed = None
    vmax = max(vmax * 1.05, 1.0)

    fig = plt.figure(figsize=(10.4, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.45, 0.72, 0.72])
    ax = fig.add_subplot(grid[0, :])
    bend = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[1, 1])
    height = fig.add_subplot(grid[2, :])
    junction = build_junction(design.junction)

    _draw_junction_electrodes(ax, junction)
    if ref_xyz is not None:
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], color="#9a9a9a", ls="--", lw=1.2, zorder=2)
    drawn = _colored_path(ax, xyz[:, :2], speed, 0.0, vmax, lw=2.4)
    fig.colorbar(drawn, ax=ax, fraction=0.025, pad=0.01, label="speed (m/s)")
    _gate_tau, gate_fraction = one_sided_time_gates(
        design.curves, design.n_time_gates, design.gate_offsets,
    )
    gate_distance = gate_fraction * distance[-1]
    gate_index = [int(np.argmin(np.abs(distance - item))) for item in gate_distance]
    ax.scatter(
        xyz[gate_index, 0], xyz[gate_index, 1], s=34, marker="D",
        facecolors="white", edgecolors="#333333", linewidths=0.9, zorder=7,
    )
    span = np.vstack([xyz[:, :2], _electrode_xy(junction, dc_only=True)])
    pad = 0.35
    ax.set_xlim(span[:, 0].min() - pad, span[:, 0].max() + pad)
    ax.set_ylim(span[:, 1].min() - pad, span[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (trap heights)")
    ax.set_ylabel("y (trap heights)")
    if title is None:
        title = "best junction and trajectory"
        if total_quanta is not None:
            title = "best design, {:.3g} quanta".format(float(total_quanta))
    ax.set_title(title)

    ax.legend(
        handles=[
            Patch(facecolor=_RF_FACE, edgecolor=_RF_EDGE, alpha=0.5, label="RF"),
            Patch(facecolor=_DC_FACE, edgecolor=_DC_EDGE, alpha=0.5, label="DC"),
            Line2D(
                [0], [0], color="#9a9a9a", ls="--", lw=1.2,
                label="nominal path" if ref_xyz is not None else "nominal waveform",
            ),
            Line2D([0], [0], color="#4C78A8", lw=1.6, label="optimized waveform"),
            Line2D(
                [0], [0], marker="D", color="#333333", ls="none",
                markerfacecolor="white", markersize=5, label="time gates",
            ),
        ],
        frameon=False, fontsize=8, loc="upper left",
    )

    _draw_junction_electrodes(bend, junction, rf_alpha=0.45, dc_alpha=0.32)
    if ref_xyz is not None:
        bend.plot(
            ref_xyz[:, 0], ref_xyz[:, 1], color="#9a9a9a", ls="--", lw=1.2,
            zorder=3, label="nominal path",
        )
    _colored_path(bend, xyz[:, :2], speed, 0.0, vmax, lw=2.0)
    bend.plot(controls[:, 0], controls[:, 1], color="#E45756", lw=0.8, ls="--", zorder=4)
    bend.scatter(fixed[:, 0], fixed[:, 1], s=26, marker="s", color="#333333", zorder=5, label="bend ends")
    bend.scatter(
        free[:, 0], free[:, 1], s=32, marker="o", facecolors="none",
        edgecolors="#E45756", linewidths=1.2, zorder=6, label="bend controls",
    )
    for index, point in enumerate(free):
        bend.annotate(
            "c{}".format(index),
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="#E45756",
        )
    bend.set_xlim(controls[:, 0].min() - 0.15, controls[:, 0].max() + 0.22)
    bend.set_ylim(
        min(-0.03, float(controls[:, 1].min()) - 0.02),
        max(0.16, float(controls[:, 1].max()) + 0.04),
    )
    bend.set_aspect("equal", adjustable="box")
    bend.set_xlabel("x (trap heights)")
    bend.set_ylabel("y (trap heights)")
    bend.set_title("bend controls")
    bend.legend(frameon=False, fontsize=8, loc="upper left")
    if ref_speed is not None:
        right.plot(ref_distance * 1e6, ref_speed, color="#9a9a9a", ls="--", lw=1.2, label="nominal waveform")
    right.plot(distance * 1e6, speed, color="#4C78A8", lw=1.6, label="optimized waveform")
    gate_speed = np.interp(gate_distance, distance, speed)
    right.scatter(gate_distance * 1e6, gate_speed, s=32, marker="D", facecolors="none", edgecolors="#333333", zorder=3, label="time gates")
    right.set_xlabel("distance along the path (μm)")
    right.set_ylabel("speed (m/s)")
    right.set_title("transport waveform")
    right.set_xlim(0.0, distance[-1] * 1e6)
    right.set_ylim(0.0, vmax)
    opt_distance, opt_height = _height_profile(design)
    if reference is not None:
        ref_height_distance, ref_height = _height_profile(reference)
        height.plot(ref_height_distance, ref_height, color="#9a9a9a", ls="--", lw=1.2, label="nominal")
    height.plot(opt_distance, opt_height, color="#4C78A8", lw=1.6, label="optimized")
    height.set_xlabel("distance along the path (μm)")
    height.set_ylabel("height (μm)")
    height.set_title("height through transport")
    height.set_xlim(0.0, opt_distance[-1])
    height.legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_transport_height(designs, out):
    """Ion height against distance along the path.

    ``designs`` is a sequence of ``(label, design, style)``. ``style`` is a
    matplotlib line-style dict. Height is a property of the path, so the
    distance is arc length and does not follow the speed waveform.
    """
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.axhline(ION_HEIGHT_M * 1e6, color="#bbbbbb", lw=0.8, zorder=0)
    for label, design, style in designs:
        distance_um, height_um = _height_profile(design)
        ax.plot(distance_um, height_um, label=label, **style)
    ax.set_xlabel("distance along the path (μm)")
    ax.set_ylabel("height (μm)")
    ax.set_title("height through transport")
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_losses(histories, out):
    """``histories`` maps a model name to the dict returned by training."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, history in histories.items():
        ax.plot(history["train"], label="{} train".format(name))
        if history.get("val"):
            ax.plot(history["val"], linestyle="--", label="{} val".format(name))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("heating-model loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_prediction(predictor, dataset, out, seed=0):
    """Predicted total heating against the simulated mean."""
    mask = dataset.successful()
    truth = total_heating(dataset.quanta[mask])
    pred = predictor.predict_heating(
        dataset.x[mask], n_samples=24, rng=np.random.default_rng(seed)
    )
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lo = float(min(truth.min(), pred["total_mean"].min()))
    hi = float(max(truth.max(), pred["total_mean"].max()))
    if hi <= lo:
        lo, hi = lo * 0.9, lo * 1.1
    yerr = np.minimum(pred["total_std"], np.maximum(pred["total_mean"], 1e-30) * 0.95)
    ax.errorbar(
        truth,
        pred["total_mean"],
        yerr=yerr,
        fmt="o",
        ms=4,
        color="#4C78A8",
        ecolor="#9ecae1",
        lw=1,
    )
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("simulated total quanta")
    ax.set_ylabel("predicted total quanta")
    ax.set_title("predictive mean")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_active(history, out):
    """Held-out error and the best simulated heating after each collection round."""
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    rounds = [row["round"] for row in history]
    axes[0].plot(rounds, [row["log_rmse"] for row in history], marker="o", color="#4C78A8")
    axes[0].set_xlabel("active round")
    axes[0].set_ylabel("held-out log10 RMSE")
    axes[0].set_title("network on the nominal neighborhood")
    axes[1].plot(rounds, [row["best_heating"] for row in history], marker="o", color="#4C78A8")
    axes[1].set_xlabel("active round")
    axes[1].set_ylabel("best simulated total quanta")
    axes[1].set_yscale("log")
    axes[1].set_title("lowest heating so far")
    for axis in axes:
        axis.set_xlim(min(rounds) - 0.3, max(rounds) + 0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_search(history, out):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    steps = [row["iteration"] for row in history]
    best = [row["best_heating"] for row in history]
    ax.plot(steps, best, marker="o", color="#4C78A8")
    ax.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best simulated total quanta")
    ax.set_yscale("log")
    ax.set_title("cross-entropy search")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
