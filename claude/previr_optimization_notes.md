# HydrodynamicTransport — PREVIR optimization branch (`previr`)

Working notes for the optimized + cleaned line of HydrodynamicTransport used by the
**PREVIR / Oyster-Rain** kernel campaign (`programming/softMode/pipeline`). Goal: run the
transport model as **few times and as cheaply as possible** to generate source→receptor
kernels for the 21 representative windows × 20 artificial tracers (one 14-day, 6-hourly
model run per window).

---

## How softMode uses this branch

- softMode declares HydrodynamicTransport as a **path dependency** — see softMode
  `Project.toml` `[sources]`: `HydrodynamicTransport = {path = "…/HydrodynamicTransport"}`.
- A path dep uses **whatever branch is checked out in this folder**. Keep **`previr`**
  checked out for PREVIR work. There is no git-rev pin — *checkout == what softMode uses*.
- Run through the **softMode environment** with **Julia 1.14 nightly** (`julia +nightly`).
  HydrodynamicTransport's *own* `Project.toml` env currently fails to precompile under this
  nightly (LibCURL / Downloads / Pkg) — a pre-existing issue — so always run via softMode's
  env, which precompiles cleanly and pulls in this working tree.

---

## Done (committed on `previr`)

### #1 — Hydro time-slab cache  *(bit-identical)*
`update_hydrodynamics!` previously re-read the entire time axis **and** both bracketing 3-D
field slabs (u, v, temp, salt, zeta × 2) from the NetCDF on **every timestep**, even though
the bracketing time index only changes when sim-time crosses a hydro output interval.
Now: the time axis is converted **once**, and slabs are read from disk **only when the
bracket index changes**, kept in a new `HydroSlabCache` on `HydrodynamicData`
(`ModelStructs.jl`, `Hydrodynamics.jl`). The 2-arg `HydrodynamicData(filepath, var_map)`
constructor is preserved, so all existing call sites are untouched.

### #2 — No per-step `deepcopy`  *(bit-identical)*
`run_simulation` snapshotted the **entire multi-tracer state** with `deepcopy(state)` every
timestep (and every CFL retry) purely to enable adaptive-dt rollback — huge per-step
allocation + GC churn. Replaced with one preallocated `work` buffer (`_copy_dynamic_state!`
= `copyto!` of the mutable dynamic fields, **no allocation**) and an **O(1) swap** to commit.
Scratch buffers (`_buffer1/_buffer2`, `flux_*`) are not copied (overwritten before read).
*Note:* `run_and_store_simulation` still uses the old `deepcopy` (not used by the campaign).

---

## Validation (must stay bit-identical)

| Script | What it checks |
|---|---|
| `validate_interp.jl` | `update_hydrodynamics!` interpolation + clamping + **non-monotonic** time with the cache (synthetic 1×1×1 NetCDF; fast) |
| `validate_optim.jl` | 1-hour run on the **real CurviLoire 2015 curvilinear grid + real NetCDF path** (#1 *and* #2 through the full loop); prints a 17-sig-fig checksum |

```
julia +nightly validate_interp.jl      # synthetic, fast
julia +nightly validate_optim.jl       # needs the local run_curviloire_2015.nc
```
Both currently pass; `validate_optim.jl` is **bit-identical** before/after the changes
(`sumA=2.17524864653079800e+05`, `sumB=8.29916896815613261e+05`, …). Always run both before
committing any further optimization. **Validate on the curvilinear + real-data path** — a
Cartesian/placeholder grid does NOT exercise #1 at all.

---

## Roadmap — speed-ups (priority order)

1. **Thread the tracer loop (#4).** The 20 passive tracers are advected serially in
   `horizontal_transport!` / `vertical_transport!` (`for tracer_name in keys(state.tracers)`).
   They're independent → `Threads.@threads` over tracers gives ~cores× on the transport step.
   Needs **per-thread flux buffers** (currently shared `state.flux_x/flux_y/flux_z`). Safe /
   bit-identical (watch FP reduction order in diffusion).
2. **Diagnose what limits `dt`** before touching the CFL barrier. Instrument
   `calculate_max_cfl_term` to report the *binding* cell/term. `dt_min=0.01` in the campaign
   smells like a few pathological thin / wetting-drying fringe cells, not the whole field — if
   so, **local subcycling** or capping velocity in sub-`D_crit`/thin cells recovers a large
   global `dt` cheaply and safely.
3. **Implicit vertical diffusion only** (if `Kz` / thin surface layers are the stiff term):
   a linear tridiagonal solve in `z`, unconditionally stable, well-posed. **Do NOT** revive
   implicit-TVD advection — it's nonlinear (limiter depends on the solution) and was already
   found unstable.
4. Reuse work across tracers within a step where geometry is shared (face areas/volumes are
   already precomputed grid data; the main shared per-step cost — the velocity field — is now
   cached by #1).

## Roadmap — cleanup ("clean it up")

- **Broken test suite:** `TestCasesModule.jl` uses a stale API (`state._buffers`, positional
  `run_simulation(grid, state, sources, ds, hydro_data, …)`, 3-arg `horizontal_transport!`),
  and `run_braided_river_test.jl` (referenced by it and `test/runtests.jl`) is **missing** —
  restore or remove. Make `test/runtests.jl` pass under the softMode env.
- **Fix HT's own `Project.toml`/`Manifest.toml`** so the package precompiles standalone under
  Julia 1.14 nightly (currently must be run via softMode's env).
- Remove the `repomix-output*.xml` source dumps from the repo root.
- Apply #2 to `run_and_store_simulation`, or document that it's unused.
- Memory: the `work` buffer ~doubles the resident `State` (same *peak* as before, now
  persistent) — fine for a workstation; revisit if RAM-bound on the full 20-tracer grid.

## Conventions

- Always run under **Julia 1.14 nightly** via the **softMode environment**.
- Every optimization must stay **bit-identical** (or be explicitly flagged as
  accuracy-affecting and re-validated against a reference kernel) — check with the two
  `validate_*.jl` scripts before committing.
