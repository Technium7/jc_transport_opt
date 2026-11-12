# Ion trap junction design with a stochastic network

This project simulates Y-junction ion transport and co-optimizes the junction geometry together with a three-dimensional trajectory. The ion starts and ends at 50 μm. The bend radius `s` (`cross_s`) builds the tangent-matched curve, and the seven bend controls are samples of that curve. Their `(x, y)` stays on the curve. Training varies the junction, the bend radius `s`, the seven control heights, and the time profile `(a, b)`. `s` and each control height stay within 15 μm of the RF center. The transport waveform is symmetric about mid-path. The search stores the four outbound gates. The center gate stays at the nominal time, and the return half is their mirror. Anomalous heating is 300 quanta/s at 1.5 MHz and scales as ω⁻². `--heating velocity` adds that energy to each mode's velocity at every step. `--heating end` accumulates the same quanta and adds them at the end of the transport.

The dynamics are the developed symplectic integrator. The RF pseudopotential comes from the junction electrodes. The default DC force is the electrode field of the least-squares voltage solution on the segmented DC electrodes: each 100 ns update is solved for the target well, then the ion is integrated in that held field. The ideal harmonic well is only an optional comparison. The stray field is off unless `--noise` is set. A network predicts the final motional quanta. A stochastic network injects input-dependent noise into its first hidden layer and is trained so that noise reproduces the spread of the simulated heating. Training is Adam followed by full-batch L-BFGS. A cross-entropy loop uses that prediction to choose which designs to simulate, fine-tunes the network the same way, and refits the sampling distribution to the best simulated designs.

## Layout

| Path | Role |
| --- | --- |
| `ion_npe/design.py` | Junction controls, bend radius, and the time profile |
| `ion_npe/junction.py` | Symmetric Y-junction and the RF Laplacian |
| `ion_npe/trajectory.py` | Round-trip path. Bend `(x, y)` comes from radius `s`; control heights are searched within 15 μm |
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

The design vector is the eight junction controls `ai, da, bi_x, bi_y, bo_x, bo_y, ci_l, co_l`, the bend radius `cross_s`, the seven bend-control heights `c0_z` through `c6_z`, and the time-profile pair `(a, b)`. Joint and nearby searches append the four outbound waveform gates. The center gate stays at the nominal time, and the return half mirrors those four. The spline uses nine interior gates. `cross_s` fixes the bend `(x, y)`. At `cross_s = 0.1` that `(x, y)` is the developed bend. The arm path and the crossing height stay at the developed design. The start and the end stay at 50 μm. Also fixed are the ion mass, RF gap and width, RF voltage and frequency, transport speed, and the target axial and radial frequencies.

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

`study` turns the stray field off. It fills a fine neighborhood of the nominal path with a Latin hypercube, and fills the wider junction and trajectory range with a second, larger hypercube. Bend radius and height stay inside the 15 μm RF-center caps. It holds some designs out and reports log10 RMSE, Gaussian NLL, and one-standard-deviation coverage for Adam and for Adam followed by L-BFGS. Active rounds then screen another space-filling pool and simulate the low-phonon designs where the network is most uncertain. `--fine` and `-n` set the two sample counts. `--iterations` is the number of active rounds.

`demo` is a short pipeline check. A real fit is `study`, or `collect` followed by `train` and `optimize`. `--lbfgs` is the number of L-BFGS iterations after Adam.

`--voltage-ns` defaults to 100. `--noise` defaults to 0, which turns the stray field off. A positive value is the standard deviation, in V/m, of the stray-field maximum on each path-frame axis. The figure from `simulate --out` draws the bend control points on the path.

## G2 cluster

Collection and the simulations inside the search run on every task of one Slurm allocation. Training stays on one task: the network is small, and the time is in the ion dynamics. Rank 0 proposes the designs, writes a batch on the shared project directory, and every node integrates its slice.

The default request matches an interactive mehta shell, repeated across 10 nodes:

```bash
srun --mem=4G --time=8:00:00 -n 1 -p mehta --pty /bin/bash
```

Slurm's `--mem` is per node, so 10 nodes at 4G is 40G. The account cap is 16 nodes and 128G total, which is 16 nodes at 8G. Do not multiply those together past 128G.

From the project directory on a G2 login node, after `mkdir -p logs` and setting `ION_PYTHON` in `cluster/env.sh` to the trapsim interpreter:

```bash
sbatch cluster/pipeline.sbatch
sbatch cluster/run.sbatch study -o outputs/study
sbatch cluster/run.sbatch joint -o outputs/joint
sbatch cluster/run.sbatch nearby -o outputs/nearby
sbatch --nodes=1 --ntasks=1 cluster/run.sbatch train -d data/train.npz -o data/models
sbatch --nodes=16 --ntasks=16 --mem=8G cluster/run.sbatch joint -o outputs/joint
```

`pipeline.sbatch` collects 400 designs, trains, then runs the cross-entropy search. `N_COLLECT`, `EPOCHS`, `LBFGS`, and `ITERS` override those counts. `study`, `joint`, and `nearby` already combine collection, training, and search, so each is one `run.sbatch` job.
