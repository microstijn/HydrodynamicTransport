# Session handoff — HydrodynamicTransport `previr` (FFSL + Float32 + sigma + vertical advection)

Continuation notes for a fresh session. Read alongside
[`previr_optimization_notes.md`](previr_optimization_notes.md) (the running optimization log — the
numbered items #1–#8 there have full detail). Branch **`previr`**, pushed to `origin/previr`
(`…ec0c924`). softMode uses this working tree via a path dep, so *checkout == what the campaign runs*.

Run standalone tests under **`julia +release`** (1.12, fast); run the real-data `validate_*.jl` /
`/tmp/*.jl` diagnostics under **`julia +nightly`** with `JULIA_NUM_THREADS=8` (they activate the
softMode env and hit the local CurviLoire 2015 `.nc`).

---

## 1. What this session shipped (8 commits, all pushed)

| commit | what |
|---|---|
| `fce6a29` | `:FFSL` conservative flux-form semi-Lagrangian advection (PPM + Zalesak FCT) |
| `4e77b08` | **fix:** MARS3D sigma vertical coordinate — physical `dz=Δσ·H0` (volumes were dimensionless) |
| `d434d4e` | **fix:** `min_depth=0.5` masks near-dry cells (tames TVD blow-up on the sigma grid) |
| `fc60f1e` | docs: the campaign `dt~1s` is a cosmetic output/receptor boundary-clamp, **not** a CFL collapse |
| `780e660` | **perf:** Float32 tracer storage (~1.5×) + `:FFSL` is now the default scheme |
| `e1a9783` | perf: precompute tracer-independent face Courant once per FFSL step (~5%) |
| `f61bea5` | perf: skip vertical advection when `w==0` (superseded by `ec0c924`) |
| `ec0c924` | **feat:** vertical advection via omega-from-continuity + implicit vertical solve |

The softMode side: `pipeline/src/C2_…jl` `ADVECTION_SCHEME="FFSL"` (committed on branch
`clean-pipeline`) and the two `K7_hydro_execution_manifest_v4.csv` copies patched `TVD→FFSL`. So the
next D1 campaign run uses **FFSL + Float32 + diagnosed vertical advection** with no further changes.

## 2. Current state of the model (what the campaign now does)

- **Horizontal:** `:FFSL` (default) — conservative, strictly positive (FCT), peak-preserving;
  gradient-CFL-limited (`dt`≈54 s median). `:TVD`/`:UP3` still available.
- **Vertical:** omega **diagnosed from continuity** (`diagnose_vertical_velocity!` in
  `Hydrodynamics.jl`, on by default; `run_simulation(...; diagnose_vertical_velocity=false)` to
  disable) + a **single implicit per-column solve** (`_advdiff_z_column!` in
  `VerticalTransportModule.jl`: backward-Euler upwind advection + CN diffusion, Thomas, allocation-free).
  Coastline/edge cells keep `w=0`. With `w≡0` it falls back to CN-diffusion-only.
- **Grid:** sigma vertical coordinate read correctly (`GridModule.jl`); `min_depth=0.5` masks
  sub-0.5 m cells as land.
- **Storage:** tracer / flux / bed-mass fields are **Float32** (`ModelStructs.FT`); arithmetic stays
  Float64. Hydro/grid fields stay Float64. `.jld2` outputs now store Float32 tracers.
- **Adaptive `dt`:** velocity-CFL bound (~54–64 s). The implicit vertical solve means the
  diagnosed `w` (which can give vertical Courant > 1 in thin cells) does **not** force `dt` down.

**Per-step cost** (real grid 467×252×10, 8 tracers, 8 threads): horizontal FFSL ~**118 ms**,
vertical ~**38 ms**, total transport ~**156 ms/step**. (Campaign = 20 tracers → ~2.5× that.)

## 3. NEXT SPEED OPTIMIZATIONS — mostly closed (see optimization-notes #7, 2026-06-13)

**A full profiling session disproved the three levers this section used to list (#1 type-stability,
#2 `@simd`/`@inbounds`, #3 pass-fusion-for-loop-overhead). Don't redo them.** The kernels are
**already type-stable** (`@code_warntype` clean — Julia specializes untyped args), and FFSL is
**memory-bandwidth-bound, not compute-bound** (the old #6 "compute-bound" claim was wrong;
contradicted #4 and is refuted by the thread-scaling curve: 8tr at 1/2/4/8 threads = 401/217/140/116
ms → 4→8 only 1.21×). Compute micro-opts therefore can't move wall-clock. Scaffolds:
`claude/audit_ffsl.jl` (type audit), `claude/bench_ffsl.jl` (timing + profile + tracer sweep),
`claude/bench_threads.jl` (thread scaling), `claude/bench_diff_ab.jl` (**same-process interleaved
A/B** — the right way to measure a change on this noisy machine; holds the old kernel locally and
compares vs the source version). **Baselines (real grid, 8 threads): pre-#9 8tr ≈ 118 ms, 20tr ≈
307 ms/step; post-#9 ≈ 113 / 294 ms; ±20% run-to-run noise — always compare best-of-N within one run.**

**Dead, measured:** type annotations (0%), `@inbounds` on the flux funcs (0% — already inlined into
an `@inbounds` loop), contiguous-Float64 Courant gather (regression), Float32 geometry
(`volume`/`face_area` → `FT`): only ~3–6% / noise-buried AND breaks the sediment `rtol=1e-9`
conservation test — because shared read-only geometry stays in L3 across tracers (not re-streamed);
the real RAM traffic is the **distinct per-tracer** arrays, already Float32 (#5).

**DONE & committed this session (branch `previr`):**
- **#9 diffusion per-line scratch** (`dc16956`): `diffuse_{x,y}!` drop the full 3-D flux buffer for
  per-line scratch (x: per-row; y: **rolling 2-row** buffer to stay i-contiguous — a naive
  per-column version regressed +17%). Bit-identical, **−14% @8tr / −18% @20tr** on the horizontal
  step (same-process interleaved A/B). All 110 tests pass.
- **cleanup** (`564c2f7`): removed the dead `fluxes_*` arg from `diffuse_{x,y}!`; FFSL skips
  `_ensure_flux_pools!` (advection + diffusion both per-line) → ~75 MB less idle memory. Bit-identical.
- FFSL full horizontal step now ≈ **113 ms @8tr / 294 ms @20tr** (was 118/307 pre-#9).

**#10 adv+diff fusion (4→2 sweeps): TRIED and REJECTED** (+7–8% regression — the per-column fused
y-sweep reads diffusion geometry strided; #9's rolling buffer reads it contiguously). Lesson:
contiguity beats pass-count here. **No cheap horizontal lever remains.** (notes #10)

**Still-untested small/clean (vertical, not horizontal):** precompute the tracer-independent vertical
`dz` (`dzc[k]=volume·pm·pn`) once per step instead of per tracer (same trick as the #6 Courant
precompute); vertical is only ~38 ms so cap ~few %.

*Not speed levers (decided):* `dt` (no headroom — `dt~1s` was a reporting artifact, `fc60f1e`);
geometry-read bandwidth (geometry caches in L3 — measured ~0%); reducing tracer count; thread
rebalance at 20 tracers (16% `wait()` is real but the busy threads are already bandwidth-throttled).

## 4. Key files / functions

- `src/HorizontalTransportModule.jl` — `:FFSL` (`advect_{x,y}_ffsl!`, `_ffsl_line!`,
  `_ffsl_ppm_edges!`, `_ffsl_face_flux[_low]`, `_compute_face_courant!`); `:TVD`/`:UP3`; the
  tracer-parallel `horizontal_transport!`. FFSL reads the precomputed Courant from `state.flux_x/y`.
- `src/VerticalTransportModule.jl` — `vertical_transport!` (implicit per-column solve);
  `_advdiff_z_column!` (adv+diff), `_cn_diffuse_column!` (diff-only); `get_dz_centers`.
- `src/Hydrodynamics.jl` — `diagnose_vertical_velocity!`; `update_hydrodynamics!` (loads UZ/VZ→u/v,
  slab cache, then diagnoses w; `diagnose_w` kwarg).
- `src/GridModule.jl` — sigma coordinate + `min_depth` masking + `volume`/`face_area` build.
- `src/ModelStructs.jl` — `const FT = Float32` (flip to `Float64` to opt out of low precision).
- `src/TimeSteppingModule.jl` — `run_simulation` (adaptive `dt`, the `diagnose_vertical_velocity`
  kwarg, output/receptor boundary clamp at L118-126 that produces the cosmetic small `dt`).

## 5. How to run / validate

- Unit tests (110): `julia +release --project=. -e 'using Pkg; Pkg.test()'`.
- Real-grid validation scripts (untracked, local; need the CurviLoire 2015 `.nc`):
  - `validate_sigma.jl` — sigma metric (`Σdz=H0`, physical volumes).
  - `validate_optim.jl` — full-loop checksum (now FFSL+F32; sanity check, not bit-identical).
  - The `/tmp/*.jl` from this session (regenerate as needed): `bench_min.jl` (per-step timing),
    `val32.jl` (FFSL mass conservation), `val_omega.jl` (diagnosed w + uniform-tracer consistency),
    `prof.jl` (alloc/GC), `bench_vt.jl` (vertical timing). Window K7_S6_W1: start `10_886_400`,
    14 days; campaign numerics TVD→FFSL, `cfl=0.9`, `dt_max=1500`, `dt_min=0.01`, `D_crit=0.05`.

## 6. Open issues / caveats (not speed)

- **Isolated horizontal distortion:** a few cells show large single-step uniform-tracer distortion
  (max ~0.98) from FFSL on the steep sigma grid — present with `w≡0` too (not the omega diagnosis).
  Same thin-cell / `min_depth` family. If it matters scientifically, options: raise `min_depth`,
  or make the omega diagnosis use FFSL's volume-weighted divergence instead of the FV `u·face_area`
  divergence (they differ where `min()` face depth ≠ cell depth — a known TVD/FFSL inconsistency).
- **Rigid-lid residual:** the omega diagnosis neglects the SSH tendency (static volumes), spread by
  layer thickness — small, inherent to the static-volume choice.
- **`min_depth=0.5`** removes 233 fringe cells (0.94%); if a different file has a pathological cell
  just above 0.5 m, TVD could still misbehave there (FFSL doesn't need `min_depth`).
- **Float32 outputs:** `.jld2` state files store Float32 tracers now — fine for the unit-release
  kernels (E-stages normalize), but a format change vs older runs.
