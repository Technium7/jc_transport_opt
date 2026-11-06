# Ion trap junction design with a stochastic network

This project simulates Y-junction ion transport and co-optimizes the junction geometry together with a three-dimensional trajectory. The ion starts and ends at 50 μm. Along the arm, through the bend, and at the crossing, the path controls are free in `(x, y, z)`. The bend controls are packed more tightly near the junction. In data collection and in the search, every free height stays within 10 μm of 50 μm, and the crossing stays within 10 μm of the origin along the +120° line. The ion's height along the simulated path is held to that same 10 μm band.

The dynamics are the developed symplectic integrator. The RF pseudopotential comes from the junction electrodes. The DC confinement is the target harmonic well along the path: it is held for 100 ns between updates, with no stray field unless `--noise` is set. A network predicts the final motional quanta. A stochastic network injects input-dependent noise into its first hidden layer and is trained so that noise reproduces the spread of the simulated heating. Training is Adam followed by full-batch L-BFGS. A cross-entropy loop uses that prediction to choose which designs to simulate, fine-tunes the network the same way, and refits the sampling distribution to the best simulated designs.

## Layout

| Path | Role |
| --- | --- |
| `ion_npe/design.py` | Junction controls plus seven bend controls `(x, y, z)` |
| `ion_npe/junction.py` | Symmetric Y-junction and the RF Laplacian |
| `ion_npe/trajectory.py` | Round-trip path. Bend controls are `(x, y, z)`, denser near the junction |
| `ion_npe/dynamics.py` | Path frame, stray field, velocity Verlet |
| `ion_npe/simulate.py` | One design in, final quanta out |
| `ion_npe/parallel.py` | Split a simulation batch across SLURM tasks |
| `ion_npe/dataset.py` | Collection of simulated designs |
| `ion_npe/model.py` | Deterministic MLP and stochastic network |
| `ion_npe/train.py` | Adam, then L-BFGS |
| `ion_npe/cem.py` | Modified cross-entropy search |
| `ion_npe/active.py` | Active collection in the low-phonon, high-uncertainty region |
| `ion_npe/study.py` | Nominal-centered data, Adam+L-BFGS, and repeated active rounds |
| `ion_npe/plotting.py` | Path, loss, and search figures |

The design vector is the eight junction controls `ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l`, then four arm controls and seven bend controls `(x, y, z)`, then `bend_z`, `cross_z`, the time-profile pair `(a, b)`, and `cross_s`. The start and the end stay at 50 μm. Also fixed are the ion mass, RF gap and width, RF voltage and frequency, transport speed, and the target axial and radial frequencies.

## Run

Use a Python environment that has the NIST `electrode` package. On the cluster that environment is the project virtualenv:

```bash
cd /share/mehta/zx296/jc_transport_opt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m ion_npe simulate --out outputs/nominal.png
.venv/bin/python -m ion_npe collect -n 40 --seeds 0 1 2 3 --workers 8 -o data/train.npz
.venv/bin/python -m ion_npe train -d data/train.npz -o data/models
.venv/bin/python -m ion_npe optimize -d data/train.npz -m data/models/stochastic.npz --workers 8
.venv/bin/python -m ion_npe study -o outputs/study
.venv/bin/python -m ion_npe demo
```

On the cluster, submit the same command as 12 tasks on the mehta node. The allocation is 48 GB and 8 hours. Rank 0 runs the command and trains. Each task simulates its own share of a collection batch:

```bash
sbatch scripts/mehta.sbatch nearby -o outputs/electrodes --noise 0 --seeds 0 -n 2000 --per-round 1000 --iterations 3 --epochs 40 --lbfgs 15
```

`--workers` is the local process pool, used when the command is not inside that step. The saved designs match a serial run with the same `--seed` when the cover is random. Training stays on rank 0: Adam, then full-batch L-BFGS.

`study` turns the stray field off. It fills a fine neighborhood of the nominal path with a Latin hypercube, and fills the wider junction and trajectory range with a second, larger hypercube. Height and crossing stay inside the 10 μm caps. It holds some designs out and reports log10 RMSE, Gaussian NLL, and one-standard-deviation coverage for Adam and for Adam followed by L-BFGS. Active rounds then screen another space-filling pool and simulate the low-phonon designs where the network is most uncertain. `--fine` and `-n` set the two sample counts. `--iterations` is the number of active rounds.

`demo` is a short pipeline check. A real fit is `study`, or `collect` followed by `train` and `optimize`. `--lbfgs` is the number of L-BFGS iterations after Adam.

`--voltage-ns` defaults to 100. `--noise` defaults to 0, which turns the stray field off. A positive value is the standard deviation, in V/m, of the stray-field maximum on each path-frame axis. The figure from `simulate --out` draws the bend control points on the path.
