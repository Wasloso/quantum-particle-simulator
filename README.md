# LiH VQE

A small Python application for studying the electronic ground-state energy of
lithium hydride with the variational quantum eigensolver (VQE). It combines
PySCF classical references, Qiskit Nature molecular models, SciPy optimization,
and optional Qiskit Aer measurements/noise.

The code is organized as ordinary functions and dataclasses. Experiments run
locally on a CPU and produce JSON files and plots.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
uv run lih-vqe --help
```

Python 3.12 is selected by `.python-version`; the package supports Python
3.11–3.13. The current lockfile was tested on Linux/WSL with Python 3.12.
Commit `uv.lock` when changing dependencies so the team uses the same versions.

## Start from the full model and compare reductions

Full LiH in STO-3G is the reference: **4 electrons, 6 spatial orbitals, 12
qubits** under Jordan–Wigner mapping. The default working calculation uses
six qubits. The smaller four-qubit model is an additional comparison.

| Molecular model | Explicit electrons | Spatial orbitals | JW qubits | Parity with two-qubit reduction |
| --- | --- | --- | --- | --- |
| Full space | 4 | 6 | 12 | 10 |
| Frozen Li core | 2 | 5 | 10 | 8 |
| Active space (2,4), default | 2 | 4 | 8 | **6** |
| Active space (2,3) | 2 | 3 | 6 | **4** |

Freezing/truncating orbitals changes the molecular approximation; reducing
symmetry qubits preserves its selected symmetry sector. Two models with ten
qubits can therefore represent different approximations. We derive the qubit
count from the physical model and mapping rather than storing it as a setting.

```bash
uv run lih-vqe models --config configs/lih.toml --output results/model-comparison.json
```

This prints the full/reduced models and their reference energy errors, without
running a VQE optimization for each. It includes the full 12-qubit Hamiltonian,
so the size/accuracy comparison does not require training its larger circuit.
The presets in this command are specifically defined for LiH/STO-3G.

There is one small base class, `ElectronicModel`, with `FullSpace`, `FrozenCore`,
and `ActiveSpace` subclasses in `models.py`. They select electrons/orbitals;
the same chemistry, VQE, and measurement pipeline handles their qubit encodings.

## Run a calculation

```bash
uv run lih-vqe run --config configs/lih.toml
```

This optimizes a 15-parameter UCCSD circuit for LiH at 1.6 Å using the default
six-qubit active model. The console reports the total energy, errors against
the active and full references, optimizer status, and physical-sector check.
A timestamped file is saved under `results/`.

For an explicit filename:

```bash
uv run lih-vqe run --config configs/lih.toml --output results/single.json
```

Existing output files are protected from overwriting. Choose a new filename
for another experiment. Result files contain configuration, library versions,
classical references, Hamiltonian terms and energy offsets, parameters and
their order, convergence history, diagnostics, and compiled gate counts.

## Scan the bond length and plot results

```bash
uv run lih-vqe scan --config configs/lih.toml --output results/scan.json
uv run lih-vqe plot results/scan.json --output results/plots
```

The default scan uses 1.2, 1.6, 2.0, 2.5, and 3.1 Å. Each point starts
independently. The JSON snapshot is updated after every geometry, and failed
calculations are recorded while the remaining geometries continue. A scan
with calculation failures exits with code 1. Interrupted scans retain their
completed points; automatic resume is not implemented yet.

Plotting reads saved results and generates `energy-curve.png` and
`convergence.png`. It does not rerun chemistry. Multiple result files can be
compared:

```bash
uv run lih-vqe plot results/scan.json results/another-scan.json --output results/comparison
```

## Example configurations

| File | Purpose |
| --- | --- |
| `configs/lih.toml` | Default six-qubit UCCSD calculation and five-point scan |
| `configs/lih-full.toml` | Full-space 12-qubit Jordan–Wigner model |
| `configs/lih-frozen-core.toml` | Freeze the Li core: 10 JW qubits (8 with parity reduction) |
| `configs/lih-six-qubit.toml` | Explicit configuration of the default (2,4) active space |
| `configs/lih-four-qubit.toml` | Smaller (2,3) active space, four parity qubits |
| `configs/lih-hea.toml` | Cheap four-qubit Ry/Rz/CX comparison circuit |
| `configs/lih-evaluation.toml` | Four-qubit ideal optimization, then noisy expectation and actual measurements |

```bash
uv run lih-vqe run --config configs/lih-six-qubit.toml
uv run lih-vqe run --config configs/lih-four-qubit.toml
uv run lih-vqe run --config configs/lih-hea.toml
uv run lih-vqe run --config configs/lih-evaluation.toml
```

`[model].kind` selects `full`, `frozen_core`, or `active`. Only the `active`
kind uses `active_electrons` and `active_orbitals`. The full-space configuration
also works with `run`, but optimizing it is substantially more expensive than
the default six-qubit experiment; inspect `models` first.

Settings omitted from a file use the defaults in `config.py`. Unknown settings
are rejected. Distances use **Angstrom**, total energies **Hartree**, and
reported errors **millihartree (mHa)**. To try another method, copy an example
and edit its `[vqe]` settings rather than changing the source.

Both optimizers have a hard `max_evaluations` limit on actual objective calls,
including finite-difference gradient evaluations. The final state is evaluated
once more for reporting. An experiment can finish normally while its optimizer
has not converged: always inspect `converged`, `message`, and `sector_valid`.
The shallow HEA is a baseline and can exhaust its budget without finding a
valid singlet; those results are marked in plots.

## What is being simulated?

The nuclei are fixed at each bond length. We estimate the electronic
ground-state energy in STO-3G, adding nuclear repulsion and inactive-electron
energy constants exactly once.

Neutral LiH has four electrons and six spatial orbitals in this basis. The
default model freezes the lowest occupied orbital and keeps two electrons in
canonical RHF orbitals **[1, 2, 3, 5]**, indexed from zero. For the supplied
z-axis geometry these include the valence occupied orbital, two virtual sigma
orbitals and one pi orbital. Parity mapping removes two redundant qubits:
eight spin-orbital modes become six qubits. The four-qubit model instead uses
**[1, 2, 5]**. Both are reduced models, not full-space LiH.

Explicit orbital indices are convenient for this first version. Recheck their
character when changing the basis or geometry range; automatic orbital
tracking is a future extension. The six-qubit model adds one orbital and
does not include the complete virtual pi pair.

Each run independently computes:

- RHF and full-space FCI with PySCF;
- active-space CASCI using the RHF orbitals;
- the exact singlet-sector eigenvalue of the mapped Hamiltonian.

The mapped reference must agree with CASCI within `1e-8` Hartree, and the mapped
HF energy must reproduce RHF before optimization starts. The classical ground
states are also checked for singlet spin. References are exact within the
chosen finite basis/model, rather than exact physical energies.

Generic circuits can change electron number and spin. The objective adds a
positive penalty for deviations from the target alpha/beta electron counts
and for nonzero total spin squared. The physical energy and penalized objective
are saved separately. `sector_valid` checks the **ideal optimized state**;
it is not a claim about its noisy execution.

The initial four-qubit LiH pilot at 1.6 Å gave approximately:

| Quantity | Energy (Hartree) |
| --- | --- |
| RHF | -7.8618647698 |
| Full STO-3G FCI | -7.8823243789 |
| Active CASCI, orbitals [1, 2, 5] | -7.8810720440 |
| UCCSD VQE | -7.8810720437 |

The circuit error is small here, while reducing the molecular model introduces
about **1.25 mHa**. The default six-qubit model reduced this to about **0.73 mHa**
at the same geometry. Reduction error varies with distance; the four-qubit
1.2 Å point exceeds the conventional 1.6 mHa threshold. Agreement with active
CASCI therefore does not establish chemical accuracy across the full curve.

## Measurements and noise

Optimization always uses exact statevector expectations. The optional
`[evaluation]` section evaluates the final parameters afterward:

- `shots > 0`: group qubit-wise commuting Paulis, rotate measurement bases,
  and sample actual Aer bitstrings. The **total** budget is split across groups.
  At least two shots per group are required. Known identity constants require
  no shots. Group-level variances account for covariance between terms.
- `noise > 0`: apply a two-qubit depolarizing channel after each compiled CX.
  A density-matrix expectation is recorded without shot uncertainty. If shots
  are also enabled, sampling uses the same noise channel.

The channel is `D_p(ρ) = (1−p)ρ + p I/4` on its two-qubit subsystem, as
parameterized by Aer. Single-qubit gates and readout are ideal. The reported
sample standard error describes sampling uncertainty, not noise bias or
molecular-model error. No readout mitigation or noisy optimization is included.

## Code structure

```text
src/lih_vqe/
  config.py       # TOML settings and validation
  enums.py        # model, mapping, ansatz, optimizer and status choices
  models.py       # ElectronicModel -> FullSpace / FrozenCore / ActiveSpace
  chemistry.py    # classical references, active Hamiltonian, physical sector
  ansatz.py       # UCCSD / HEA circuits and common compilation
  vqe.py          # SciPy optimizer and exact objective evaluations
  evaluation.py   # grouped measurements and depolarizing noise
  experiments.py  # one run, sequential scan, JSON snapshots
  plotting.py     # figures from saved results
  cli.py          # command-line interface
```

To add a circuit family, extend `build_ansatz()` and the configuration choices.
To add an experiment, compose these functions in `experiments.py`. A future
dashboard can call `run_experiment()` or read the same JSON files; the numerical
modules do not depend on a UI framework.

This first version supports neutral, closed-shell diatomics. Operators stay
sparse, and reference diagonalization is restricted to the physical sector:
full 12-qubit LiH has only 225 determinants at fixed alpha/beta populations.
H₂ is used for integration tests. Full-space FCI remains a small-system reference and is not
intended to scale to large molecules/bases. Compilation uses a fixed
`rz/sx/x/cx` basis and all-to-all connectivity; physical routing is a later
experiment.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests include real LiH/H₂ calculations, sector-restricted references, mapping
agreement, the optimizer budget, analytical measurement/noise cases, and a
run/save/plot integration check. Use `uv run pytest -m 'not integration'` for
just the lightweight checks. Generated results are ignored by Git.

Method background: [VQE](https://arxiv.org/abs/1304.3061),
[four-qubit LiH experiment](https://arxiv.org/abs/1704.05018),
[Qiskit Nature](https://qiskit-community.github.io/qiskit-nature/), and
[PySCF CASCI/FCI](https://pyscf.org/user/mcscf.html).
