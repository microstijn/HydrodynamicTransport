# Breathing-sigma — session handoff (continue here)

Self-contained continuation notes. Branch **`previr`** (HydrodynamicTransport). The breathing-sigma
continuity correction is **complete and validated end-to-end** (forward magnitude fix + reverse-time
adjoint). This session's job: two **refinements** (machine-precision reciprocity; wet/dry parking),
then **Phase 5** (kernel re-baseline). Read alongside `SESSION_HANDOFF.md` (the older FFSL/opt notes)
and the PoC oracle `breathing_sigma_poc/README.md`.

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
- **Reverse-time reciprocity** (adjoint duality ⟨Ma,b⟩ vs ⟨a,M*b⟩ on a deep interior support) →
  **0.40% rel err** (within the FCT-limiter floor).

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

## 2. IMPROVEMENT 1 — machine-precision reciprocity (unlimited-linear mode)

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

## 3. IMPROVEMENT 2 — wet/dry parking

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
   → 137/137 (sanity).
2. Re-run `claude/breathing_sigma_poc/production_reciprocity.jl` → 0.40% (confirms the baseline).
3. **Improvement 1** (unlimited-linear): vet the multi-cell donor transpose with an agent →
   implement `linear` flag → reciprocity ≤1e-12.
4. **Improvement 2** (parking): vet the parking projection with agents → implement opt-in `parking`
   flag (dynamic active set) → C≡1 + Mc-drop + wetting-cycle conservation + intertidal reciprocity.
5. **Phase 5**: thread `breathing`(+`linear`) into softMode C2 manifest → cluster re-baseline →
   re-certify (burden should be unchanged; magnitudes corrected).

**Process reminder (this project's rule, `[[use-math-agents-for-math-fixes]]`):** spawn several
independent math agents to vet NEW math BEFORE writing production code, reconcile, then checkpoint.
It caught `Hn→min(Hn,Hnp)` and the `Vd`-not-`Va` framing this session, and the mid-read replay fix
(reciprocity 76%→0.40%). The AUP content-filter trips on "virus/norovirus" in spawned-agent prompts —
use NoV/tracer/pathogen-marker (`[[agent-prompt-virus-filter]]`).
