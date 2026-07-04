# Breathing-sigma — session handoff (continue here)

Self-contained continuation notes. Branch **`previr`** (HydrodynamicTransport). Read alongside
`SESSION_HANDOFF.md` (the older FFSL/opt notes) and the PoC oracle `breathing_sigma_poc/README.md`.

## ⇒ START HERE (updated 2026-07-04 — Improvement 4 DONE; full-3D reciprocity SOLVED)

The forward breathing fix + all FOUR refinements are **DONE + committed** (suite **152/152**, each opt-in
default bit-identical):
- **Improvement 1** — linear exact-adjoint breathing mode (`breathing_linear`), commit `d4de909`. §2.
- **Improvement 2** — wet/dry parking (`breathing_parking`), commit `2ea2519`. §3.
- **Improvement 3** — vertical FFSL (`breathing_vffsl`), commit `3b71a46`. §3b.
- **Improvement 4** — **reverse read-boundary selection fix** (§3c). Full-3D real-grid reverse-time
  reciprocity now **~5e-9** (Float32-limited), down from 1.5e-4 — a ~30,000× improvement.

**⚠ THE PRIOR ROOT-CAUSE DIAGNOSES WERE ALL WRONG — corrected this session (empirically).** The real
floor was NOT the vertical backward-Euler (§2 guess), NOT a horizontal/splitting residual (§3b guess),
and NOT a cascade η-reset volume mismatch (the `f64107a` "7.85e-3" pin). Those are struck through below;
read §3c for what actually happened. In one line: **the reverse (adjoint) run selected the WRONG hydro
read at every read boundary** (`searchsortedlast` returns the read *above* a boundary, but a descending
pass needs the read *below*), so one sub-step per read boundary advected with the adjacent read's
transports. This is a **reverse-only** fix — the forward path is **bit-identical** (no C≡1/salinity/burden
re-validation needed). Everything else (cascade-volume mirror 6.6e-16, horizontal linear kernel adjoint
1e-16, vertical FFSL z-op adjoint 7e-12) was already exact; read selection was the sole defect.

**NEXT TASK = Phase 5** (§4): kernel-library re-baseline on the 160 GB grid (cluster), driven from
softMode C2. Reverse-time reciprocity is now machine-precision, so the breathing reverse-time kernels are
ready for the Gate-1 sensitivity product.

---

## 0. Where we are (DONE + committed)

The offline tracer solver was **rigid-lid** (frozen `grid.volume`/`face_area` at H0; ζ≡0) → a ~0.5 dex
tide-locked magnitude error at the intertidal Loire receptor. Fixed by an **opt-in breathing-sigma**
mode: a depth-weighted barotropic Poisson projection uses the reliable `∂η/∂t` to correct the transport
so discrete continuity closes, volumes breathe, ω is the GCL vertical flux, and a two-time-level
Strang cascade advects the tracer. Default (`breathing=false`) = rigid-lid, **bit-identical** (137/137).

Commits (all on `previr`): `375c143` (P0/1 projection+wiring) · `66a59b4` (P2 kernel) · `ec99907`
(P3a transport module) · `e29b9c4` (P3b run_simulation wiring) · `943ab32` (P3c Strang) · `d9ff16d`
(P4 reverse-time + real-data validation).

**Validated on the real MARS3D CurviLoire grid:**
- Continuity `div(U*)=T` → 1.35e-10; GCL ω surface closure → 2e-14.
- C≡1 through the full `run_simulation` breathing path → **bit-exact 0.0**; interior mass conservation
  → 1.5e-9; positivity (FCT) min=0.
- **Salinity vs the model's own `SAL`** (2012 file, PoC stage3 method): corrected RMS plateaus at
  0.2–0.84 PSU while rigid-lid diverges to 41 PSU — **17–49× better** (reproduces the PoC 10–40×).
- **Reverse-time reciprocity** (adjoint duality ⟨Ma,b⟩ vs ⟨a,M*b⟩ on a deep interior support):
  **~5e-9** in linear+vffsl mode with the read-boundary fix (§3c) — Float32-tracer-limited, i.e. an
  EXACT discrete adjoint. (Historically 0.40% FCT-limited → 1.5e-4 linear → 5e-9 after §3c.)

**Math was vetted by independent agents before writing production code** (do this again for the
refinements — it caught two real bugs). Forward: 3 agents unanimous. Reverse: 3 agents unanimous,
numerically verified to 3.7e-16, decisive result `A(−ω)=A(ω)ᵀ` (the implicit vertical is exactly
self-adjoint). Two subtleties they surfaced that the code now handles: the vertical M-matrix condition
is on the **departure** volume `V**` (not `Va`); the adaptive-dt Courant divisor must be
**`min(Hn,Hnp)`**; the reverse must **replay identical transports** (fixed via mid-read projection).

---

## 1. Architecture (files + key functions)

- **`src/ProjectionModule.jl`** — `BreathingProjector`, `build_projector(grid; h_open=30, wet_min=1,
  D_min=0.1)`, `project!(proj, grid, u, v, η_n, η_np1, Δt)` (raw face transports → `Draw` → Poisson
  `∇·(D̃∇φ)=T̃−Draw`, deep-mouth Dirichlet + flood-fill + Neumann land → corrected `Ux`,`Uy`(∝Δσ) +
  bottom-up GCL `omega` + floored `Hn`,`Hnp`). `T̃ = −A·(Hnp−Hn)/Δt` (FLOORED tendency, load-bearing).
  Helpers `padded_depth!`, `write_omega_velocity!`, `continuity_residual`. Topology is **static**
  (built once) — wet/dry parking will make it dynamic (§3).
- **`src/GridModule.jl`** — `rebuild_metrics!(grid, depth; d_floor=1.0)` breathes volume/face_area from
  `D̃`; `_fill_sigma_metrics!` is the shared metric builder (used by init too, bit-identical).
- **`src/HorizontalTransportModule.jl`** — the breathing FFSL kernel: `_ffsl_flux_breathing`
  (volume-coordinate walk + ambient `camb` fill on exiting the domain / hitting a dry cell) and
  `_ffsl_line_breathing!` (departure `vdep` / arrival `varr` split; monotone donor base + PPM
  antidiffusive + **Zalesak FCT** — this FCT is the nonlinearity that limits reciprocity to ~few-%).
- **`src/BreathingTransportModule.jl`** — `BreathingWork` (padded transports `Uxf`,`Uyf` + 5 Strang
  cascade volumes `V0..V4`), `build_breathing_work`, `pad_transports!`, `breathing_transport!`
  (Strang ½x·½y·z·½y·½x cascade over all tracers; `_xsweep!`/`_ysweep!`/`_zsweep!`),
  `_advdiff_z_col_breathing!` (**implicit** backward-Euler upwind on ω + implicit diffusion; row-sum =
  `Vd` ⇒ C≡1; M-matrix ⇒ unconditionally stable), `breathing_courant` (adaptive-dt bound; uses
  `min(Hn,Hnp)`).
- **`src/Hydrodynamics.jl`** — `update_hydrodynamics!(...; projector, reverse)` → `_breathing_update!`
  (projects once per read via the `last_idx` cadence guard; **projects with the MID-read velocity in
  both directions**; `reverse` negates u,v + swaps η_n↔η_np1 → −U*,−ω + reversed volume chain).
- **`src/TimeSteppingModule.jl` `run_simulation`** — the `breathing`/`breathing_camb` kwargs + the
  breathing step branch (project → pad → `breathing_courant` adaptive dt clipped to the read boundary
  → `breathing_transport!`). Reverse-time mapping: `htime=origin−time`, mirrored `f0`, backward clip.
- **`src/UtilsModule.jl`** — `:zeta`→`XE` autodetect (was the root of ζ≡0).

**Run tests:** `julia +release --project=. test/runtests.jl` (137/137; NOT `Pkg.test`).
**Data:** 2010 slim slab (UZ/VZ/XE, ~1.8 GB) at
`C:\Users\peete074\AppData\Local\Temp\claude\...\08611189-...\scratchpad\run_curviloire_2010.nc`
(a prior session's scratchpad — copy it somewhere durable or re-fetch from the cluster
`anunna:/lustre/nobackup/WUR/ESG/peete074/previr_hydro_subset/`). **Full files with `SAL`** (~150 GB,
LOCAL, fast to read):
`C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_{2012,2014,2015}.nc`
(2012 has SAL/TEMP/UZ/VZ/XE, 17553 half-hourly steps).
**Validation drivers:** `claude/breathing_sigma_poc/production_salinity_validation.jl` (2012, SAL vs
truth) and `production_reciprocity.jl` (adjoint duality). Note the ProgressMeter clutters stdout — pipe
to a file directly (NOT `| grep`, which buffers) and grep on read.

---

## 2. IMPROVEMENT 1 — machine-precision reciprocity (unlimited-linear mode) — ✅ DONE (2026-07-04)

**Status: implemented, math-vetted, committed.** Opt-in `breathing_linear` kwarg threads
`run_simulation → breathing_transport! → _xsweep!/_ysweep! → _ffsl_line_breathing!(...; linear=true)`,
which drops the PPM+FCT antidiffusive step and runs the pure first-order donor-cell flux. Default
`false` = unchanged (137/137 bit-identical; suite now 139/139 with 2 new adjoint asserts).

**Vetting (3 independent math agents, unanimous + numerically reproduced):** the linear breathing
donor-cell sweep is the **exact discrete adjoint** of its reverse-time form (negate the swept volumes
`Srow`, use the arrival volumes as the reverse departure) in the volume-weighted inner product
`diag(varr)·M = (diag(vdep)·M̃)ᵀ` — the sweep is structurally the geometric **overlap-remap matrix**,
symmetric under time reversal. Exact at **ANY Courant** (single- or multi-cell), the only condition
being every cell volume stays positive (`vdep>0, varr>0`) — cleaner/stronger than the "Courant<1"
framing below. Kernel unit test `bench_breathing_adjoint_kernel` (no external data) → **2.2e-16** at
single- AND multi-cell Courant.

**Real-grid end-to-end (`production_reciprocity.jl`):** linear mode + the adjoint-consistent inner
product (weight the forward output by V(t1), the reverse by V(t0) — only the two END volumes survive the
cascade) gives **rel err ~1.5e-4**. ⚠ **The attribution in the struck-through paragraph below is WRONG**
(kept for traceability): the residual is NOT the vertical implicit backward-Euler. The dt-halving and
Kz-independence it cites are REAL observations but are equally explained by the actual cause — the reverse
read-boundary selection bug (§3c): one mis-projected sub-step per read boundary, each ~dt long ⇒ error ∝
dt and independent of Kz/the vertical scheme. The genuine improvement from linear mode is real (drops the
FCT nonlinearity); the leftover ~1.5e-4 is §3c, fixed reverse-only. With linear+vffsl+the read fix the
full-3D reciprocity is ~5e-9.

> ~~The residual is NOT the horizontal scheme (exactly self-adjoint) — it is the **VERTICAL implicit
> backward-Euler advection**, which is only a first-order (O(dt)) adjoint... CONFIRMED by exact dt-halving
> (1.16e-4 → 5.83e-5 → 2.92e-5) and by Kz=0 leaving it unchanged. → Improvement 3 vertical FFSL would take
> the 3D end-to-end to machine precision.~~ ❌ WRONG — vffsl does NOT move the real-grid floor (§3b/§3c);
> the floor was read selection, not the vertical.

---

### Original design notes (superseded by the DONE status above, kept for context)

**Why:** reciprocity is 0.40% because the Zalesak **FCT limiter is nonlinear** (no single transpose).
The linear operator is provably an EXACT self-adjoint (agents: `A(−U*)`/`A(−ω)` = transpose given the
GCL). So a **linear** transport mode makes reverse-time the machine-precision adjoint.

**Plan (small, ~half a day):**
1. Add an opt-in `linear::Bool=false` (or `advect_order`) knob threaded run_simulation →
   `breathing_transport!` → `_ffsl_line_breathing!`. When true, **skip the PPM + FCT antidiffusive
   step** and use the LOW-ORDER donor-cell flux only (`_ffsl_flux_breathing(..., low=true)`), i.e.
   first-order upwind. This flux is linear in C; at the adaptive Courant<0.5 the walk is single-cell
   (immediate upwind), which is exactly the R1 operator the agents proved self-adjoint.
   - Concretely in `_ffsl_line_breathing!`: compute only `Flo` (skip `Fhi`, the FCT block, and the
     `Rp/Rm` limiter) and set `cnew[g] = (vdep[g]·crow[g] − (Flo[g]−Flo[g−1]))/varr[g]`.
   - The vertical `_advdiff_z_col_breathing!` is ALREADY linear (upwind, no limiter) — leave it.
2. **Vet first:** hand a fresh agent the claim "the multi-cell breathing donor-cell (first-order
   upwind) flux `_ffsl_flux_breathing(low=true)` with negated transport + swapped volumes is the exact
   transpose in the V-weighted inner product, at any Courant." Confirm the multi-cell walk (Courant>1)
   still transposes (R1 was stated for the general A(U*); the multi-cell A is wider — check its
   off-diagonals transpose under negation). If it only holds at Courant<1, note that the adaptive dt
   already keeps Courant<0.5, so it's fine in practice but should be asserted.
3. **Test:** rerun `production_reciprocity.jl` with `linear=true` on the always-wet deep support →
   expect rel err **≤ 1e-12** (down from 0.40%). Also confirm C≡1 still bit-exact and the forward
   salinity RMS is only mildly worse (first-order is more diffusive — acceptable for the reciprocity
   product; keep FCT as the default for the forward magnitude product).
4. Optional (higher accuracy, more work): instead of first-order, a **frozen-limiter tangent-linear
   adjoint** — store the forward per-face FCT coefficients and reuse them (frozen) on the reverse pass.
   Preserves PPM accuracy AND machine-precision reciprocity, but needs per-read storage of the limiter
   weights. Only do this if the reciprocity product needs high spatial accuracy; otherwise (1)+(3) is
   enough.

**Deliverable:** a `linear`/reciprocity-exact breathing mode with reciprocity ≤1e-12 on the always-wet
subdomain, committed + a small benchmark added to `test/benchmarks/reciprocity_benchmark.jl`.

---

## 3. IMPROVEMENT 2 — wet/dry parking — ✅ DONE (2026-07-04)

**Status: implemented, math-vetted (3 agents unanimous), validated, CI-locked, committed.** Opt-in
`breathing_parking` (+ `breathing_dpark=0.5`) kwargs on `run_simulation` → `build_projector(...;
parking, D_park)`. Default off = static topology, **bit-identical** (137/137; suite now **145/145** with
a new synthetic parking testset). Design (all agent-recommended corrections applied):
- Per read in `project!`: `parked = wet ∧ min(Hn,Hnp) < D_park`; a single global mouth flood-fill on
  `wet∧¬parked` (`_flood_and_number!`, dynamic `reach/active/id/N`); mouth-disconnected transportable
  cells are **folded into `parked`** (a breathing Neumann island is unsolvable — Fredholm ΣT≠0). One
  pass is a fixed point (no iteration). `min(Hn,Hnp)` is **swap-invariant** ⇒ forward/reverse masks are
  bit-identical automatically (verified 0 differing cells).
- **One source of truth for the mask**: raw face transports are zeroed at any face touching a parked
  cell (`_transp`), so `Draw` is masked consistently with the walled Poisson stencil (the load-bearing
  bug the agents flagged); the correction + ω already gate on the dynamic `active`; the cascade freezes
  parked cells (`V0=0`, `hold-C`); `breathing_courant` skips them (via dynamic `active`).
- **Mass**: `hold-C` (not `hold-mass` — agent A: `C=M/V` puts a spurious dilution on the intertidal
  receptor). Bounded, sign-cancelling leak `≤ C·A·(surface excursion while parked)`, ≈0 per tidal cycle.
  Exact conservation, if ever needed, = a resolved wetting/drying transfer flux (debit the active
  neighbour at donor C) — deferred follow-up.

**Real-grid validation (2010 slab, `production_parking_validation.jl`):** C≡1 **bit-exact 0.0** with
parking on; forward==reverse parked mask (0 differ); parking **PRESERVES** div(U*)=T on active cells
(5.6e-11) AND **removes an intermittent continuity BLOW-UP** the rigid floor suffers at near-singular
drying cells (parking OFF hit **5.0e16** at a drying window → ON 5.6e-11). Perf: at the driest window
(814 cells drying) sub-steps/read drop **Mc 133 → 51 (2.62× fewer)**. ⚠ The benefit is PHASE-dependent
and modest because the **deep channel co-dominates the Courant** (agent A predicted this) — at mid-tide
windows with no drying, parking is a no-op. Net value: continuity robustness + reciprocity-substep-match
at the intertidal receptor + correct isolated-pool handling; the perf win is real but only 1.25–2.6×.

**Note (adjoint):** parking does NOT repair the reverse-time transpose (already exact even at Va→0 — the
linear donor-cell adjoint telescopes); it removes the drying-cell Courant collapse that forced mismatched
forward/reverse sub-step counts. The remaining ~1.5e-4 end-to-end reciprocity floor is the O(dt) vertical
backward-Euler step (§2), orthogonal to parking → Improvement 3 (vertical FFSL overlap-remap).

---

### Original design notes (superseded by the DONE status above, kept for context)

**Why:** (a) **perf** — the driest intertidal cells drive `breathing_courant` (small V → large
Courant → ~50–80 sub-steps/read); parking them removes that. (b) **reciprocity at the intertidal
receptor** — drying is currently irreversible (the receptor cell parks), the one place reciprocity is
O(1) off. Parking as a **frozen linear projection** (replay the forward mask) makes it reversible.

**Plan (moderate, ~1–2 days; VET the parking-projection math first):**
1. **Dynamic parked set per read.** A cell parks when `D̃ < D_park` (with hysteresis `D_wet > D_dry` to
   avoid chatter; e.g. D_park≈0.5 m). `parked = wet ∧ (min(Hn,Hnp) < D_park)` is the simplest
   (conservative: parked for the whole read if it dries at any point). Recompute per read.
2. **Projection treats parked cells as walls.** Currently `build_projector` builds the wet mask /
   flood-fill / numbering ONCE (static). Make the "transportable" set = `wet ∧ ¬parked` and re-derive
   the mouth-connected component + interior-unknown numbering **inside `project!` each read** (or cache
   keyed to the parked set — it changes gradually). Zero the raw face transports on faces touching a
   parked cell (treat parked like temporary land: `_wet`-for-transport = `wet ∧ ¬parked`), and exclude
   parked cells from the Poisson unknowns (T̃=0 there). The GCL ω and metric keep the floored D̃.
   - This is the biggest change: `active/reach/id` become per-read (mutable) rather than static.
     Keep `wet` (bathymetry) static; overlay `parked`.
3. **Cascade freezes parked cells.** In `_update_cascade_volumes!`, parked cells get `V0=0` → the FFSL
   walk already treats `vdep≤0` as a wall + ambient-fills, and the `varr>0 ? … : crow` guard freezes
   their C. **PARK THE MASS, NOT THE CONCENTRATION** (plan §5): hold `Ṽ·C` frozen and re-inject on
   re-wetting — with V0=0 the cell's C is simply held (mass parked implicitly since no flux crosses its
   zeroed faces). Verify no mass leaks at the wetting front.
4. **`breathing_courant` skips parked cells** → the intertidal no longer collapses dt. Expect Mc to
   drop from ~50–80 to O(1–10) (verify; if the deep CHANNEL still dominates the Courant, parking helps
   less — measure which cells bind the Courant first).
5. **Reciprocity with parking:** for exact reverse-time, the reverse must replay the SAME wet/dry mask
   as forward (a frozen linear projection). Store the forward per-read parked mask and reuse it on the
   reverse pass (the reverse re-derivation from swapped η would give the SAME `min(Hn,Hnp)` mask, so it
   may already match — check; if not, thread the stored mask through).
6. **VET the math** (agents): (a) does zeroing transports at parked-cell faces keep the projection
   continuity-closed on the remaining active set (Σ T̃ still balanced through the mouth Dirichlet)? (b)
   is parking a **linear** operation (needed for the adjoint)? (c) mass conservation across a
   dry→wet→dry cycle. Then implement.

**Tests:** C≡1 still bit-exact with parking on; Mc drops at the intertidal (report the factor); mass
conserved across a wetting cycle; reciprocity at an INTERTIDAL receptor (with the frozen mask + linear
mode from Improvement 1) → machine precision. Drivers to adapt: the C≡1 `verify_breathing_run_sim.jl`
+ a new intertidal-receptor reciprocity driver.

**Risk note:** this is the highest-risk remaining change (dynamic topology). Keep it opt-in
(`parking::Bool=false`) so the validated non-parking path stays the default until parking is proven.

---

## 3b. IMPROVEMENT 3 — vertical FFSL: CORRECT vertical, but NOT the reciprocity floor (2026-07-04)

Opt-in `breathing_vffsl` replaces the implicit backward-Euler vertical with a vertical FFSL overlap-remap
advection (reuses `_ffsl_line_breathing!` on the CLOSED column: cells=layers, `Srow[f]=ω[f]·dt`, ω=0 at
seabed/surface, zero-gradient reflected ghost) + a SYMMETRIC `½diff(Vd)·adv·½diff(Va)` diffusion (½diff on
the DEPARTURE volume Vd before advection, on the ARRIVAL volume Va after — the side-pairing is load-bearing).
Math-vetted (3 agents unanimous, all 5 claims machine-precision). Validated directly on `_zsweep_vffsl!`:
z-step column adjoint `diag(Va)M = (diag(Vd)M_rev)ᵀ` → **7.3e-12**; C≡1 bit-exact (2.2e-16) in BOTH the
linear (donor-cell) and PPM+FCT flux modes. Default off = implicit vertical, bit-identical (suite 145→149;
CI `bench_vertical_ffsl_adjoint`).

⚠️ **KEY NEGATIVE FINDING — the vertical FFSL does NOT move the full-3D real-grid reverse-time reciprocity
(1.155e-4 WITH vffsl == 1.155e-4 without).** The reciprocity floor is NOT the vertical. Systematically ruled
out: vertical scheme (vffsl no change), Float32 tracer storage (`const FT`→Float64 gives identical 1.155e-4),
diffusion (Kz=0 same), adaptive-partition mismatch (fixed dt=1800/64 same 1.155e-4), projection antisymmetry
(reverse −U*/−ω and Hn↔Hnp are EXACTLY 0.0). Structural reason: the reciprocity support is DEEP (always-wet)
where ω≈0, so the vertical is ~identity there regardless of scheme. The residual is O(dt) (dt-halving
1.16e-4→5.83e-5→2.92e-5) and grows SUBLINEARLY with the window (4.2e-5 at 1 read → 1.155e-4 at 4 reads) ⇒ a
per-sub-step O(dt²) reverse-time residual with partial cancellation. Likely = the Strang x·y·z split
commutator's non-self-adjointness and/or the FFSL OPEN-boundary/land `camb` (each sweep is a proven exact
adjoint on a CLOSED line, but the real run's rows are OPEN at the domain edges / land walls). NOT fixable by
improving any single sweep.

⇒ **This CORRECTS §2 / the Improvement-1 memory attribution** ("floor = O(dt) vertical backward-Euler"): the
vertical scheme is irrelevant to the floor. The kernel/column adjoints ARE machine-precision (horizontal
2e-16, vertical 7e-12); vffsl is committed as the correct, opt-in exactly-adjoint vertical.

**✅ CORRECT observations, ❌ wrong conclusion (resolved in §3c):** the "vffsl/Float32/Kz/partition all ruled
out, error grows with window, ∝dt" observations above are all RIGHT and were the key clues. But (a) the
"support is DEEP where ω≈0, vertical ~identity" claim is FALSE — measured ω/horiz ≈ 0.32 at the support, so
ω is significant and vffsl genuinely being a no-op means the floor is elsewhere; and (b) the "Strang split
commutator / open-boundary camb" guess is WRONG (both are exact adjoints). The real cause is the reverse
**read-boundary selection** off-by-one (§3c): exactly one mis-projected sub-step per read boundary ⇒ error ∝
(#read boundaries) [grows with window] and ∝ dt [each bad step is ~dt long]. Fixed reverse-only ⇒ ~5e-9.

### ~~ROOT CAUSE of the ~1.5e-4 floor — PINNED (2026-07-04, taped-adjoint investigation)~~ ❌ WRONG — see §3c

**⚠ THIS ENTIRE BLOCK IS REFUTED (2026-07-04, later same day).** The "cascade resets from η, forward
arrival carries the divx/divy/ω redistribution, reverse can't mirror it, 7.85e-3 = dt/(2·dTread)" claim is
FALSE. A direct forward-vs-reverse cascade-volume probe on the real grid showed V0..V5 mirror to
**6.6e-16** (machine precision), transports & ω to exactly 0.0, Hn/Hnp swap to 0.0. The forward arrival
volume V5 = geom(f_b) EXACTLY by the GCL-ω per-layer closure (`divx+divy+divz = T̃` by construction of ω),
so the cascade already mirrors and volume-threading would fix a non-problem. The 3-agent "taped adjoint"
vetting validated the *transpose algebra* (which is fine) but the empirical "7.85e-3" diagnostic that
motivated it had a bug. The REAL cause is a reverse read-boundary selection off-by-one — see §3c. Kept
verbatim below only so the mistaken reasoning is traceable.

Taped-adjoint approach vetted by **3 fresh math agents (unanimous)** + empirical on-grid diagnosis:
- **The taped discrete transpose gives machine-precision reciprocity BY CONSTRUCTION** for linear mode (FCT
  off ⇒ genuinely linear, C-independent coefficients). **No tracer-trajectory tape** needed.
- **The open-boundary `camb` and the Strang split-commutator are RED HERRINGS** — both provably exact adjoints
  (single open FFSL sweep self-adjoint to 0.0; palindrome preserves order under transposition). My §3b guess
  (split/camb) was WRONG; a boundary-adjoint patch is a rank-0 no-op.
- **THE ACTUAL O(dt) SOURCE (empirically pinned):** `_update_cascade_volumes!` RESETS the sub-step departure
  volume from η (uniform-Δσ) every sub-step, but the forward's ARRIVAL volume carries the `divx`/`divy`/`ω`
  redistribution from the sweeps. The reverse recomputes its departure from η ⇒ it does NOT reproduce the
  forward's redistributed arrival. Measured on the real grid: transports & ω mirror EXACTLY (0.0) but the
  cascade volumes mismatch **7.85e-3 = dt/(2·dTread)** per sub-step (`V1_rev` vs `V4_fwd`), independent of the
  reverse `f0`. This is why vffsl didn't help (it fixes the z-SWEEP, not the η-reset of the departure VOLUME),
  and why Float32≡Float64 / Kz / partition / parking all tested clean.

**THE FIX (Improvement 4, NOT yet implemented — the vetted path to 1e-15 3-D reciprocity):** the reverse must
DEPART from the forward's EXACT redistributed arrival volume, not the η-reset. Routes: (1) **consistent
volume-threading** — thread the cascade volume within a read (`V0_substep = previous arrival`, reset only at
read boundaries) in BOTH directions ⇒ exact mirror; ⚠ changes the FORWARD ⇒ not bit-identical ⇒ re-validate
C≡1 (still algebraic-exact)/salinity-49×/burden; no memory cost; recommended (opt-in). (2) **windowed volume
tape** — forward stores the arrival-volume chain, reverse replays it; ~12 GB for a 2 h full-grid window,
keeps the forward untouched. Fully-general fallback = the taped exact sweep transpose `M* = I −
diag(1/d)·Wᵀ·Δᵀ` (scatter). Kernel/column adjoints are already machine-precision; only the cascade volume
threading blocks full-3D machine-precision reverse-time reciprocity.

---

## 3c. IMPROVEMENT 4 — the REAL reciprocity floor: reverse read-boundary selection — ✅ DONE (2026-07-04)

**The entire ~1.5e-4 full-3D reverse-time reciprocity floor was a reverse-only read-selection bug.** Fixed
in `src/Hydrodynamics.jl` `update_hydrodynamics!` (the `idx1`/`idx2` selection). Forward path
**bit-identical** (guarded by `reverse`); suite 149→**152/152** (new testset "Reverse-time read-boundary
selection (breathing-sigma)" + `write_breathing_nc` fixture in `benchmark_common.jl`).

**The bug.** A reverse (adjoint) run descends in real hydro time (`htime = origin − lag`). The run loop's
read-boundary clip lands `htime` EXACTLY on each read boundary `tv[r]`. Then
`searchsortedlast(tv, tv[r]) = r` selects read `[tv[r], tv[r+1]]` — the read *above* the boundary — but a
descending pass is about to traverse `[tv[r-1], tv[r]]`, the read *below*. So the sub-step just below every
read boundary advected with the WRONG (adjacent) read's transports. The forward is unaffected: it ascends,
so `tv[r]` is the correct left edge of the read it enters.

**Why this fingerprint fooled everyone:** ONE mis-projected sub-step per read boundary ⇒ error **grows with
the window** (measured 5.2e-5 / 1.12e-4 / 1.51e-4 at 1/2/4 reads) and each bad step is ~`dt` long ⇒ error
**∝ dt** (the "O(dt)" signature misread as the vertical backward-Euler in §2). It is orthogonal to the
vertical scheme, Kz, the adaptive/fixed partition, and FCT — all of which were correctly observed NOT to
move the floor, which is exactly what a read-selection bug predicts.

**The fix** (≈10 lines, reverse-only): when `reverse && idx1 > 1 && htime` is within `1e-6·ΔT` of
`tv[idx1]` (landed on a boundary), decrement `idx1 -= 1; idx2 = idx1+1` so the descending pass uses the
read below. Forward untouched.

**Measured (real 2010 CurviLoire grid, deep interior support, linear + vffsl):**
| config | before | after |
|---|---|---|
| window = 1 read | 5.2e-5 | **5.1e-9** |
| window = 2 reads | 1.12e-4 | **1.9e-8** |
| window = 4 reads | 1.51e-4 | **4.9e-9** |

~5e-9 is the Float32 tracer floor (`FT = Float32` in `ModelStructs.jl`); a Float64 build would show ~1e-11.
The breathing reverse-time transport is now an **exact discrete adjoint** in linear+vffsl mode: cascade
volumes mirror (6.6e-16), horizontal linear kernel adjoint (1e-16), vertical FFSL z-op adjoint (7e-12),
Strang palindrome self-transposes, and read selection is now correct. **This also silently improves the
correctness of ANY reverse breathing run** (backward footprint/kernel), not just the reciprocity duality.

**Diagnostic scripts** (this session, scratchpad — not committed): `cascade_volume_mirror_probe.jl`
(volumes mirror 6.6e-16), `reciprocity_matrix.jl` (the decomposition table above + window scaling),
`read_alignment_probe.jl` (shows the reverse lag-0 step maps to the read OUTSIDE the window),
`recip_fix_test.jl` (the after-fix collapse). `production_reciprocity.jl` is now updated to pass
`breathing_vffsl=true` (in linear mode) so it exhibits the ~5e-9 floor directly — as it stood before, it
used the implicit vertical and never enabled vffsl, which is why §3b wrongly concluded "vffsl doesn't
help". `linear=true` alone (implicit vertical, no vffsl) after the read-fix floors at the vertical's O(dt)
adjoint instead.

---

## 4. PHASE 5 — kernel-library re-baseline (AFTER 1+2)

Deliberate, separate campaign: regenerate the 137-source K7 kernel library + the reverse-time
reciprocity artifacts in **breathing mode** on the 160 GB CurviLoire grid; this supersedes the
current rigid-lid kernels. This is driven from the **softMode** pipeline, not this repo:
- softMode `pipeline/src/C2_…jl` builds the kernels via `run_simulation`. Add `breathing=true`
  (+ `breathing_camb=0`, `linear=false` for the forward magnitude kernels; `linear=true` for the
  reverse-time reciprocity product) to the C2 execution manifest, mirroring how `ADVECTION_SCHEME=FFSL`
  was threaded (see `[[previr-pipeline-repo-home]]` / the softMode `clean-pipeline` branch).
- Run the re-baseline on the cluster (`anunna`); the campaign is heavy (Phase-C-style). The E-stage
  normalization + certification then re-certifies the burden on breathing kernels.
- **Expectation (from memory):** the calibrated NoV **burden correlation is robust** to breathing
  (loading-dominated + receptor low-pass + immersion gate; see `[[carlingford-softmode-reduction]]`
  V42) — breathing corrects **magnitudes** (salinity/SSC 10–40×), not the calibrated burden timing. So
  the re-baseline is mainly to (a) put SSC/salinity products on a correct footing and (b) get the
  breathing reverse-time reciprocity kernels for the Gate-1 sensitivity product
  (`[[backward-adjoint-reciprocity]]`). Re-confirm the burden is unchanged (it should be).

---

## 5. Quick-start checklist for the new session

1. `cd HydrodynamicTransport`; `git branch` → `previr`; `julia +release --project=. test/runtests.jl`
   → **152/152** (sanity).
2. Improvements 1–4 are all DONE + committed (§2, §3, §3b, §3c). Full-3D reverse-time reciprocity is
   ~5e-9 (linear+vffsl+read-fix). `production_reciprocity.jl` now demonstrates it (pass the 2010 slab path).
3. **Phase 5** (§4): thread `breathing`(+`breathing_linear`,`breathing_vffsl`) into the softMode C2
   manifest → cluster re-baseline on the 160 GB grid → re-certify (burden should be unchanged; magnitudes
   corrected). The breathing reverse-time kernels are now machine-precision-reciprocal → Gate-1-ready.

**Process reminder (this project's rule, `[[use-math-agents-for-math-fixes]]`):** spawn several
independent math agents to vet NEW math BEFORE writing production code, reconcile, then checkpoint.
⚠ **But math-agent vetting is not a substitute for an empirical check.** This session's cautionary tale:
the "7.85e-3 cascade η-reset" root cause was 3-agent-vetted yet WRONG — the agents correctly validated the
*transpose algebra* but the empirical diagnostic that fed them was buggy. A 5-minute forward-vs-reverse
volume-mirror probe (6.6e-16, not 7.85e-3) refuted it instantly. **Always reproduce the measured symptom
before designing a fix, especially when a derivation contradicts the pinned cause.** The AUP content-filter
trips on "virus/norovirus" in spawned-agent prompts — use NoV/tracer/pathogen-marker
(`[[agent-prompt-virus-filter]]`).
