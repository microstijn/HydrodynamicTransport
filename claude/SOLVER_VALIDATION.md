# Numerical validation of the HydrodynamicTransport tracer solver

**Status:** complete, 2026-06-18. Branch `previr`. All studies reproducible from the repo root with
`julia +release --project=. validate_advection.jl` (and `validate_vertical.jl`, `validate_diffusion.jl`),
and a fast subset runs in `Pkg.test()` (127/127 pass).

## Why

Every PREVIR exposure kernel inherits the numerics of this offline tracer advection–diffusion solver,
but the solver had only ever been checked for *internal consistency* (mass-conservation rtol, the
MARS3D sigma `Σdz=H0` fix in `validate_sigma.jl`, a bit-identical optimisation checksum in
`validate_optim.jl`, 110 unit tests). It had never been run against cases whose exact behaviour is
known — the standard way to validate such a solver. This adds that layer: analytical benchmarks with
quantitative error metrics and order-of-accuracy studies for all three transport operators
(horizontal advection, vertical advection, diffusion), comparing the three advection schemes.

## Method

Each operator is **isolated** and driven by a prescribed analytical field / initial condition, then
compared to the exact solution. To isolate horizontal advection through the real `run_simulation`
path, the previously **dead** `Kh`/`Kz` keyword arguments (hard-coded to 1.0 / 1e-4 in the
FFSL/TVD/UP3 branch) were threaded through `horizontal_transport!`/`vertical_transport!`; the defaults
are unchanged so all existing callers stay bit-identical, while `Kh=0` now yields pure advection.
Velocity is supplied as a staggered analytical field via a small NetCDF (`test/benchmarks/`); the
non-sigma grid gives a unit-height column (`dz=1/nz`) for the vertical cases. Tracers are stored in
Float32, so absolute error floors near ~1e-7 — amplitudes are kept O(1).

Harness: `test/benchmarks/benchmark_common.jl` + `advection_benchmarks.jl`, `vertical_benchmarks.jl`,
`diffusion_benchmarks.jl`. Metrics: volume-weighted L1/L2/L∞ vs exact, relative mass drift,
global min/max (positivity / over-undershoot), peak retention, and convergence order `p` from a
log–log fit of L2 vs grid spacing.

## Results

### Group A — horizontal advection (`:FFSL` production, `:TVD`, `:UP3`)

| Case | Scheme | key metric | result |
|---|---|---|---|
| A1 translation, order | FFSL | p(L2) | **2.02** (monotone PPM → 2nd order at the smooth peak) |
| A1 translation, order | TVD  | p(L2) | 0.84 |
| A1 translation, order | UP3  | p(L2) | **diverges** (L2 grows with refinement) |
| A2 Gaussian rotation (1 rev, Co 0.5) | FFSL | L2 / peak-ret / min / mass-drift | 1.2e-3 / **0.96** / **0.0** / −1e-5 |
| A2 rotation | TVD | peak-ret / min | 0.52 (diffusive) / −6e-15 |
| A2 rotation | UP3 | min / max | −1067 / +1071 (**unstable**) |
| A2 rotation, **Courant 3** | FFSL | L2 / peak-ret | 5.5e-4 / **0.98** (stable at Co≫1) |
| A3 Zalesak slotted cylinder | FFSL | min / max | **0.0 / 0.998** (no under/overshoot) |
| A3 Zalesak | TVD | min / max | −5e-12 / 0.68 (clips the disc) |
| A3 Zalesak | UP3 | min / max | −3.7e5 / +3.8e5 (**unstable**) |

**Takeaway:** FFSL is conservative (mass drift ~1e-5–1e-8, Float32-limited), strictly positive,
peak-preserving, ~2nd-order on smooth fields, and stable at Courant > 1 — justifying it as the default.
TVD is monotone but markedly diffusive; **UP3 (legacy, explicit 3rd-order upwind) is unstable for
sustained advection** and should not be used.

### Group B — vertical transport (diagnosed omega + implicit upwind/CN)

| Case | metric | result |
|---|---|---|
| B1 continuity closure (diagnosed w) | max relative net-flux divergence | **1.1e-16** (machine zero) |
| B1 same metric with w ≡ 0 | — | 0.31 (uncompensated HDiv → diagnosis is necessary) |
| B2 1-D vertical advection | p(L2) | **0.91** (≈1, implicit upwind is 1st order) |
| B2 | min over run | ≥ 0 (positive) |
| B3 stability at vertical Courant 5 | finite / min / max | true / **0.0** / **0.70 ≤ 1.0** (bounded, no blow-up) |

**Takeaway:** `diagnose_vertical_velocity!` closes the discrete volume budget to machine precision for
a depth-integrated non-divergent flow (so a uniform tracer is preserved); the implicit vertical solve
is 1st-order accurate and unconditionally stable where an explicit upwind would blow up.

### Group C — diffusion (explicit horizontal, Crank–Nicolson vertical)

| Case | metric | result |
|---|---|---|
| C1 2-D horizontal diffusion | p(L2) / σ²-relerr / mass-drift | **1.99** / <7e-7 / ~1e-7 |
| C2 1-D vertical CN diffusion | p(L2) / σ²-relerr | **1.94** / <3e-3 |
| C3 CN stability, diffusion number 10 | finite / bounded | true / yes (unconditionally stable) |

**Takeaway:** both diffusion operators are 2nd-order and recover the analytical Gaussian-spreading
variance; CN vertical diffusion is unconditionally stable.

## For the paper

Proposed subsection "Numerical validation of the transport solver": the three tables above (condensed)
+ figures from `test/benchmarks/make_figures.jl` (**CairoMakie**, kept out of the package deps):
L2-vs-spacing convergence for advection/vertical/diffusion, and a rotation/Zalesak scheme comparison.
Results CSVs:
`advection_validation_results.csv`, `vertical_validation_results.csv`, `diffusion_validation_results.csv`.

## Limitations / honest notes

- FFSL convergence is ~2nd (not 3rd) order: the monotonicity (Colella–Woodward + Zalesak FCT) limiter
  clips the smooth Gaussian peak, the usual order reduction for a positivity-preserving scheme.
- Mass-conservation drift is Float32-storage-limited (~1e-5–1e-8 relative), not Float64 machine zero.
- Group B/C vertical cases run on the non-sigma unit column (`dz=1/nz`); the sigma-grid physical-`dz`
  path is covered separately by `validate_sigma.jl`.
- Not covered here (future): the LeVeque deformational-swirl test; comparison against MARS3D **native**
  tracer output on the real CurviLoire grid.
