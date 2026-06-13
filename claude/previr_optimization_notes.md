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

### #3 — Vertical CN diffusion: factorize once per call  *(bit-identical)*
`vertical_transport!` rebuilt and **re-factorized an identical tridiagonal Crank-Nicolson
operator** (`A \ rhs`) for *every* water column, *every* tracer, *every* step. On the
curvilinear grid `grid.z_w` is spatially uniform → the operator is column-independent. Now
`B` and `lu(A)` are built **once per call** and reused across all columns via `mul!`/`ldiv!`,
and the per-column slice-copies were replaced with views (`VerticalTransportModule.jl`).
Numerically identical (same `A`, same `B`, same LU path). Also threaded the previously-serial
`diffuse_x!`/`diffuse_y!` over vertical layers and removed boundary-face array-literal
allocations in the TVD advection (`HorizontalTransportModule.jl`).

**Measured on the real CurviLoire grid (467×252×10), 20 tracers, 8 threads:**
`vertical_transport!` **1993 → 218 ms/step (9.1×)** — the original actually *slowed down* with
threads because per-column allocation caused GC contention; the factorize-once version scales.
Total transport (`horizontal_transport!` + `vertical_transport!`) **~2568 → ~731 ms/step (3.5×)**.
`validate_optim.jl` checksum unchanged; 81/81 tests pass. (Committed: `adbe669`.)

*Tried and reverted:* rebalancing the horizontal threading from layers (`k`, ~10) to rows
(`j`/`i`, ~hundreds) made it ~2× **worse** — arrays are column-major so a `k`-layer is one
contiguous block; threading over `k` keeps each thread on a contiguous slab.

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

### #4 — Tracer-level threading  *(bit-identical)*
Transport now threads over the (independent) **tracers** instead of inside each tracer:
`horizontal_transport!` and `vertical_transport!` run a `Threads.@threads` loop over tracer
chunks, each task using its own scratch flux buffers. Horizontal uses lazily-allocated
per-task flux pools on `State` (`flux_x_pool`/`flux_y_pool`, scratch → not copied in
`_copy_dynamic_state!`); vertical uses tiny per-task column scratch. The inner TVD advection /
diffusion kernels are now serial (`:ImplicitADI` keeps its own internal threading and stays on
the serial-tracer path).

**Measured (real grid, 20 tracers, 8 threads):** transport **740 → 618 ms/step (1.2×)** over
#3. Only ~1.2× — not the ~2× hoped — because the transport is **memory-bandwidth-bound** at 8
threads (the kernels stream large arrays; fewer barriers + better balance help, but the memory
wall dominates). Bit-identical; 81/81 tests pass. Cumulative transport speedup vs the original
baseline: **~2568 → ~618 ms/step (4.2×)**.

*Caveat:* adding fields to `State` means restarting from a `.jld2` checkpoint written by an
*older* struct may fail (regenerate checkpoints). `run_and_store_simulation` still `deepcopy`s
the whole `State` per step, which now also copies the flux pools — another reason to migrate it
off `deepcopy` (or leave it; it's not on the campaign path).

## Advection schemes — `:FFSL` (opt-in high-fidelity)

A third horizontal advection scheme alongside `:TVD` (default) and `:UP3`: **`:FFSL`**, a
conservative **flux-form semi-Lagrangian** (Lin–Rood) sweep with monotone **PPM**
reconstruction + **Zalesak FCT** limiter (`HorizontalTransportModule.jl`:
`_ffsl_ppm_edges!`, `_ffsl_face_flux[_low]`, `_ffsl_line!`, `advect_{x,y}_ffsl!`; wired into
the tracer-parallel `horizontal_transport!`). Select with `advection_scheme = :FFSL`.

**Properties (validated):**
- **Conservative** — face fluxes telescope; machine-precision mass conservation on a uniform
  grid, FP-limited ~1e-9 on curvilinear (the `C ↔ mass` `÷V`/`×V` round-trip, since `C` not
  `C·V` is reconstructed to stay monotone where `V` varies).
- **Strictly positive / monotone** — FCT blends a donor-cell base with the PPM antidiffusive
  correction; degrades gracefully to donor-cell where the gradient-CFL is violated.
- **Peak-preserving** — retains sharp plumes ~1.5× better than TVD.
- **Correct open-boundary outflow**; dry/land faces (mask + `D_crit`) are blocked (zero flux).

**Stability limit is the velocity-gradient (Lipschitz) CFL, not the advective CFL.** FFSL is
stable at large advective Courant, but adjacent departure points must not cross:
`calculate_max_gradient_cfl_term · dt < 1` (`|∂u/∂x|`, `|∂v/∂y|` per direction, dimensional
split → the *max* of the two, not the sum). The adaptive controller
(`TimeSteppingModule.run_simulation`) automatically uses this term for `:FFSL`. Recommended
use: `use_adaptive_dt = true` + `advection_scheme = :FFSL` to auto-pick the largest safe `dt`.

**NOT a speedup on coastline/estuary grids.** On CurviLoire 2015 the gradient-CFL is *smaller*
than the advective CFL (median ratio ~0.5 — UZ/VZ have strong cell-to-cell structure), so FFSL
needs ~2× more steps **and** costs ~1.7× more per step. It buys **fidelity, not speed**, there.
Use it when peak/shape accuracy matters; keep `:TVD` for the campaign's cheap-kernel goal.

## Roadmap — speed-ups (priority order)

1. **`dt` is the next lever, not threading.** Transport is memory-bandwidth-bound, so further
   thread work won't help much. **Diagnose what limits `dt`** before touching the CFL barrier. Instrument
   `calculate_max_cfl_term` to report the *binding* cell/term. `dt_min=0.01` in the campaign
   smells like a few pathological thin / wetting-drying fringe cells, not the whole field — if
   so, **local subcycling** or capping velocity in sub-`D_crit`/thin cells recovers a large
   global `dt` cheaply and safely.
2. **Implicit vertical diffusion only** (if `Kz` / thin surface layers are the stiff term):
   a linear tridiagonal solve in `z`, unconditionally stable, well-posed. **Do NOT** revive
   implicit-TVD advection — it's nonlinear (limiter depends on the solution) and was already
   found unstable.
3. Reuse work across tracers within a step where geometry is shared (face areas/volumes are
   already precomputed grid data; the main shared per-step cost — the velocity field — is now
   cached by #1).

## Cleanup — done (committed on `previr`)

- **Test suite rewritten.** `test/runtests.jl` is now a single self-contained suite (synthetic
  in-memory grids + tiny NetCDF fixtures, no external data/network) covering the current API:
  flux limiters, `lonlat_to_ij`, state init, grid geometry, curvilinear-from-NetCDF + autodetect,
  hydro interpolation **+ slab cache (#1)**, vector rotation, Cartesian mass bounds, end-to-end
  curvilinear `:TVD`/`:UP3` runs, adaptive `dt`, source/sink + decay, sediment settling/bed
  exchange, and receptor monitoring. **81 tests pass** via `julia +release` (1.12) `Pkg.test()`.
  The stale `TestCasesModule.jl` and the never-existed `run_braided_river_test.jl` dependency are
  gone.
- **src/ is library-only.** Removed the 7 stale dev scripts; moved
  `run_loire_simulation_with_oysters.jl` → `examples/`. Removed the dead `ext/CairoMakieExt.jl`
  (was never wired into `[weakdeps]`/`[extensions]`).
- **Deps pruned.** Dropped `Revise`, `BenchmarkTools`, `UnicodePlots` (only the deleted scripts
  used them); moved `Test` to `[extras]`/`[targets]`. Package precompiles cleanly under 1.12.
- **Repo hygiene.** Removed all `repomix-output*.xml` dumps; rewrote the incoherent `.gitignore`.
- **Bug fixed.** `flush_receptor_monitor!` shadowed `Base.values` with a local of the same name,
  so every CSV flush threw `UndefVarError` — fixed; now exercised by the receptor testset.

## Roadmap — cleanup (remaining)

- **Standalone precompile under Julia 1.14 nightly** still fails (LibCURL / Downloads / Pkg) — a
  nightly-specific issue, unchanged by the above. The package precompiles + tests cleanly under
  1.12 release; nightly runs continue via the softMode env.
- Apply #2 (no per-step `deepcopy`) to `run_and_store_simulation`, or document that it's unused.
- Memory: the `work` buffer ~doubles the resident `State` (same *peak* as before, now
  persistent) — fine for a workstation; revisit if RAM-bound on the full 20-tracer grid.

## Conventions

- Always run under **Julia 1.14 nightly** via the **softMode environment**.
- Every optimization must stay **bit-identical** (or be explicitly flagged as
  accuracy-affecting and re-validated against a reference kernel) — check with the two
  `validate_*.jl` scripts before committing.
