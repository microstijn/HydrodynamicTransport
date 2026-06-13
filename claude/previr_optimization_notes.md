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
| `validate_sigma.jl` | MARS3D **sigma vertical-coordinate** correctness (real grid): sigma detected, `Σdz=H0`, physical `dz`/volumes, short run finite + conserving |

```
julia +nightly validate_interp.jl      # synthetic, fast
julia +nightly validate_optim.jl       # needs the local run_curviloire_2015.nc
```
Always run before committing any further **optimization** — those must stay bit-identical.
**Validate on the curvilinear + real-data path** — a Cartesian/placeholder grid does NOT exercise
#1 at all. The `validate_optim.jl` reference checksum was **reset by the MARS3D sigma fix** (a
deliberate correctness change — see below); current baseline `sumA=2.12878484367176134e+05`,
`sumB=4.69499076685949112e+05` (sigma fix + `min_depth=0.5`). The pre-sigma value was
`sumA=2.17524864653079800e+05`, `sumB=8.29916896815613261e+05` (that grid had wrong,
dimensionless cell volumes).

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

## Correctness fix — MARS3D sigma vertical coordinate  *(NOT bit-identical — intentional)*

The CurviLoire campaign file is **MARS3D**: it has **no** ROMS `s_w`/`Cs_w`/`hc`, only a CF
`ocean_sigma_coordinate` (`level`/`SIG`, layer **centres** in `[-1,0]`, uniform Δσ=0.1) plus
bathymetry `H0` and SSH `XE`. `initialize_curvilinear_grid` previously only knew the ROMS form,
so it fell back to `z_w=[-1…0]` → **dimensionless `dz=0.1`**. Cell volumes were therefore
~`depth`-times too small (and spatially undistorted), and the vertical CN diffusion used the
wrong (dimensionless) `dz`. **Fixed** (`GridModule.jl`, `VerticalTransportModule.jl`,
`SettlingModule.jl`, `UtilsModule.jl`, `SourceSinkModule.jl`):

- `GridModule` detects the sigma coordinate (`_autodetect…` → `:sigma_center`), builds the
  dimensionless sigma **interfaces** from the centres (`_sigma_centers_to_interfaces`), and scales
  the metrics by the **per-column depth**: physical thickness `dz(i,j,k)=Δσ_k·H0(i,j)` [m].
  Volumes are now in **m³** and vary with depth. **Decision: depth = `H0` only** (static; SSH `XE`
  averages to ~0 over a tidal run, and a time-varying volume would need ALE / a volume-flux term —
  out of scope). Non-sigma (ROMS/synthetic) grids are byte-for-byte unchanged (`is_sigma=false`).
- Physical `dz` is recovered everywhere as `volume·pm·pn` (no new grid field; one definition in
  `get_dz_centers`); all `z_w`-difference `dz` look-ups were switched to it.
- **Face** thickness uses `min` of the two column depths (conservative; zero over land → no-flow at
  the coast; keeps `face_area/volume ≤ pm` so the `u·pm` advective-CFL estimate stays valid — an
  *averaged* face depth amplifies flux into thin cells at steep slopes and **NaNs**).
- **Dry cells keep a floored (1 m nominal) volume** so `volume>0` everywhere — several kernels
  divide by `volume` unguarded (e.g. `advect_x/y_tvd!`) and relied on the old never-zero volume.
- The vertical CN factorize-once (#3) **could no longer be shared** (α now per-column), so it is
  now an **allocation-free per-column Thomas solve** (`_cn_diffuse_column!`). The 9.1× win came
  from killing per-column *allocation*, not from sharing the LU, so most of it is retained.

**Validation:** `validate_sigma.jl` (real MARS3D grid) — sigma detected, `Σ_k dz = H0` to machine
precision, `dz` spatially variable 0.1–8.6 m, volumes 10³–10⁷ m³, short run finite + conserving.
110/110 unit tests pass (new `MARS3D sigma vertical coordinate` testset, incl. `min_depth`).
**`validate_optim.jl` checksum intentionally changes** (the old numbers were wrong); new corrected
baseline (sigma + `min_depth=0.5`): `sumA=2.12878484367176134e+05 sumB=4.69499076685949112e+05`.

### Follow-up — near-dry cells break explicit TVD  *(fixed: `min_depth`)*

Correct (depth-scaled) volumes exposed a latent **TVD positivity weakness**: in the thin upstream
channel, the small shallow-cell volumes make the explicit `dt/volume` advection term lose
positivity, producing unbounded ± oscillations that the open-boundary feedback then re-injects —
**non-conservation by ~230×** (`validate_optim` tracer B: `maxC~3e10`, `minC~-1.8e10`, total mass
230× the injected mass). Diagnosis: **smaller `dt` makes it *worse*** (so it's not a CFL limit —
it's the per-step `1/V` amplification compounding), and **`:FFSL` is unaffected** (conservative +
positive by construction: `maxC~5e4`, `mass/inj=0.58`, same as the healthy deep tracer A).

**Fix:** `initialize_curvilinear_grid(...; min_depth=0.5)` (default **0.5 m**) — cells with
`0 < H0 < min_depth` are masked out as **land** (zero-depth → mask off, zero-area faces via the
`min` face depth, floored volume), so no flux enters them. On CurviLoire this removes 233 fringe
cells (0.94 %, all `< 0.5 m` intertidal flats a static-volume model can't represent anyway) and
restores TVD to `mass/inj=0.58` (matching FFSL / tracer A), `maxC~5e4`. `min_depth=0.0` opts out
(raw `H0>0` mask). Thresholds 0.2 m still blow up (the binding cell is `H0=0.46 m`); 0.5 m is the
first that catches it — **`:FFSL` needs no `min_depth`**, so it stays the robust choice for thin
estuary channels. (Plausibly the same pathological-thin-cell family behind the campaign `dt~1s`,
Roadmap #1.)

## #5 — Float32 tracer storage  *(memory-bandwidth lever; ~1.5x)* + `:FFSL` as default

Transport is memory-bandwidth-bound, so the lever is **bytes streamed per step**, not the scheme
or `dt` (which is fixed at ~60 s, velocity-CFL-bound — see RESOLVED above; there is no `dt` lever).

**Float32 tracer storage** (`ModelStructs.FT = Float32`): the per-tracer arrays streamed every step
— `tracers`, `_buffer1/2`, `flux_x/y/z`, `flux_*_pool`, `bed_mass` — are stored in Float32, halving
their memory traffic. **Arithmetic stays Float64** (kernel scratch + intermediates promote; only
storage is reduced). Hydro/environment fields (`u,v,w,zeta,T,S,tss,uvb`) and all grid metrics stay
Float64. Tracer-array function signatures relaxed `Array{Float64,3}` → `AbstractArray{<:Real,3}`.

- **Speed (real grid, 8 tracers, 8 threads):** per-step horizontal `TVD 162→105 ms (1.54x)`,
  `FFSL 172→125 ms (1.38x)`, vertical `→38 ms`. Not the full 2x because the F64 velocity/metric
  arrays are still streamed.
- **Accuracy (real FFSL run, sources incl. far-upstream):** F32 vs an F64 build agree to **~1e-6
  relative** (mass/inj identical to 5 digits, maxC/sumC to ~6 digits); strictly positive in both.
  Negligible for unit-release kernels. Flip `FT = Float64` in `ModelStructs.jl` to opt out.
- **Caveat:** `.jld2` state outputs now store Float32 tracer fields (downstream E-stages read +
  normalize, so fine, but it's a format change). `validate_optim.jl` is no longer bit-comparable
  to the F64 baseline (F32 by design) — treat it as a sanity check, not a bit-identical regression.

**`:FFSL` is now the default `advection_scheme`** in `run_simulation` / `run_and_store_simulation`
/ `estimate_stable_timestep` (was `:TVD`). Benchmarked at only **~1.26x TVD end-to-end** here
(per-step 1.06x — the extra PPM+FCT work hides under the bandwidth wall — × dt 1.19x from the
slightly tighter gradient-CFL), and it is conservative + strictly positive + peak-preserving,
whereas TVD produced large negative undershoots (`minC~-3.5e8`) at the sharp upstream releases.
Combined, **`:FFSL`+Float32 per-step (125 ms) is *faster* than the old `:TVD`+Float64 (162 ms)**;
end-to-end ≈ 0.92x of TVD+F64 with strictly better numerics. NOTE: the D1 campaign passes
`advection_scheme` explicitly from the execution manifest (still `TVD`); to adopt `:FFSL` for the
campaign, set that column to `FFSL` in `K7_hydro_execution_manifest_v4.csv` — the code default only
affects callers that don't specify a scheme.

### #8 — Vertical advection via omega-from-continuity + implicit vertical solve  *(physics)*

The CurviLoire MARS3D file stores **no vertical water velocity** (only `UZ`/`VZ` horizontal,
`TEMP`/`SAL`, sediment concentrations, and `WS_Mud` = mud *settling* velocity). So the offline
transport had **no vertical advection** — and on a sigma grid the horizontal sweeps alone don't
preserve a uniform tracer. Now `diagnose_vertical_velocity!` (`Hydrodynamics.jl`, called from
`update_hydrodynamics!`; toggle with `run_simulation(...; diagnose_vertical_velocity=…)`, default
**on**) reconstructs omega from continuity using the horizontal transports already loaded:
per column `Wflux(k+1)=Wflux(k)-HDiv(k)` with **rigid-lid closure** (w=0 at seabed & surface, the
residual column divergence — the neglected SSH tendency, since volumes are static — spread by layer
thickness), then `w=Wflux/area`. **Coastline/boundary cells are left at w=0** (a land/edge neighbour
breaks horizontal continuity → the "divergence" is an artefact; diagnosing there pumped spurious
vertical transport and lost ~24% of an upstream tracer's mass — e.g. cell (411,72) at the
min_depth-masked river head).

`vertical_transport!` was reworked: when w is active it now does a **single implicit per-column
solve** — **backward-Euler upwind advection + CN diffusion** (`_advdiff_z_column!`, allocation-free
Thomas). Implicit because the diagnosed omega gives vertical Courant > 1 in thin cells, where the
old **explicit** upwind is unstable and the adaptive controller (horizontal-CFL only) never catches
it — that was losing/redistributing mass. Implicit upwind is **unconditionally stable, conservative
(flux-form), and positivity-preserving**. With w≡0 it falls back to the pure CN-diffusion path.

**Validation (real grid):** diagnosed w is physical (max 9.6e-3, median 6.6e-5 m/s); mass conserved
(`up2/nw/ind` all back to ≈ released fraction); strictly positive; uniform-tracer distortion ~**1.9×**
lower than w≡0. Cost: vertical ~38 ms with advection vs ~38 ms diffusion-only — the implicit
advection adds terms to the same Thomas solve, so it's **nearly free**; ~+14 ms vs the old
diffusion-only-no-advection (the price of the missing physics). 110/110 tests pass. *Caveat:* a few
isolated cells still show large single-step uniform-tracer distortion (max ~0.98) from horizontal
FFSL on the steep sigma grid — present with w≡0 too, i.e. not caused by the diagnosis.

### #6 — FFSL face-Courant precompute  *(~5%; FFSL is compute-bound)*

The per-face Courant numbers are **tracer-independent** (function of velocity / free-surface /
geometry), but the original FFSL recomputed them inside every tracer's sweep. Now
`_compute_face_courant!` fills them **once per step** into the otherwise-unused `state.flux_x` /
`flux_y` scratch (Float32), and `advect_{x,y}_ffsl!` read that instead of recomputing. ~5% per-step
(real grid, 8 tracers: ~125 → ~118 ms); conservation + positivity unchanged. NOTE: this couples the
FFSL kernels to the precompute — direct callers (e.g. unit tests) must call `_compute_face_courant!`
first; `horizontal_transport!` does it automatically.

**Why only ~5%, and why an F32-volume cache gave ~0%:** after Float32 tracers, FFSL is
**compute-bound in its FCT/PPM limiter** (Zalesak min/max + divisions, parabola reconstruction),
not bandwidth-bound on the geometry reads. So read-reduction tricks plateau. The real per-step
lever was already taken (Float32 tracers, #5). Further speed would need cutting the limiter
arithmetic (risky, touches the numerical core — not worth it) or the cheaper non-positive `:TVD`.
Bottom line: ~5% banked; FFSL's cost is the price of its positivity + peak fidelity.

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

## RESOLVED — the campaign `dt~1s` was a reporting artifact, not a CFL collapse

Replicated the K7 2015 campaign (D1 numerics: TVD, `dt_init=20`, `cfl=0.9`, `dt_max=1500`,
`dt_min=0.01`, `dt_growth=1.1`, `D_crit=0.05`; V10 sources incl. the far-upstream river cells) on
`run_curviloire_2015.nc` for a small tracer subset. Findings (`tmp_cfl_scan` over all four 2015
windows S1_W2/S2_W2/S6_W1/S6_W2 + a live run):

- **Velocity CFL never forces small `dt`:** floor **39–52 s**, median 60–69 s, *never <30 s*,
  binding on normal H0≈2.4–3.6 m cells. `min_depth` doesn't change it.
- **The adaptive controller can't collapse `dt` on tracer instability** — its CFL term
  (`calculate_max_cfl_term`) reads only `u,v`, which transport never modifies, so even a TVD
  blow-up is *accepted* at ~60 s.
- **The tiny `dt` is the boundary clamp** (`TimeSteppingModule.jl` ~L118-126): each step is
  shortened to land exactly on `end_time`, the next **full-state output** time, and the next
  **receptor-monitor** time. The campaign writes 6-hourly output and records receptor kernels
  **hourly**, so once an hour `dt` is cut to a small remainder. The progress bar's
  `min_timestep_s` then shows that remainder (live run with hourly output: `min_timestep_s=2.6`
  vs `max_timestep_s=59.17`; over 336 hourly boundaries in 14 days the min remainder → ~1 s).
- **It costs ~nothing:** ~336 short remainder-steps among ~20 000 normal ~60 s steps (the clamp
  replaces a step, it doesn't add one). So `dt~1s` is **cosmetic** — there is no `dt` speed lever
  here. (If the cosmetic min ever matters, align the output/receptor cadence to a multiple of the
  working `dt`, or have the controller absorb the remainder into the prior step.)

## Roadmap — speed-ups (priority order)

1. **No `dt` lever** (see RESOLVED above): the working `dt` is already ~60 s, velocity-CFL-bound on
   ordinary cells. Transport is memory-bandwidth-bound, so further thread work won't help much.
   A real speed-up would need a scheme change (e.g. `:FFSL` only helps where its gradient-CFL
   exceeds the advective one — not on this estuary) or coarser output. Not worth chasing now.
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
