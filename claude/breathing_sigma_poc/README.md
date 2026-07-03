# Breathing-sigma continuity-correction — PoC oracle scripts

Validated proof-of-concept for the breathing-sigma fix (see the production plan:
`~/.claude/plans/inhydrodynamictransport-i-gt-omments-iridescent-teapot.md`). These standalone scripts
prove the scheme on the real MARS3D CurviLoire field and are the **verification oracle** for the
production integration. Run with `julia +release -t auto --project=<HydrodynamicTransport>`.

They need a local MARS3D slab: the 2010 slim subset (`run_curviloire_2010.nc`, UZ/VZ/XE, ~1.8 GB; on the
cluster at `anunna:/lustre/nobackup/WUR/ESG/peete074/previr_hydro_subset/`) and the full 2012 file (has
`SAL` ground truth; local at `Documents/PREVIR_PROJECT/01_raw/hydro/CurviLoire/run_curviloire_2012.nc`,
150 GB). Fix the `path` constants at the top of each script.

| script | what it does | validated result |
|---|---|---|
| `real_probe.jl` | rigid-lid C≡1 probe (the DEFECT baseline) | receptor C≡1 error: peak factor ~10, rectified ~0.52 dex |
| `diag_engine_div.jl` | engine divergence vs −Area·∂η/∂t (why rigid-lid fails) | r≈−0.10, div 16× the barotropic signal |
| `stage1_projection.jl` | depth-weighted barotropic Poisson projection (SparseArrays) | continuity closes to **5e-10**; \|u'\|/\|u\| median 0.21 |
| `stage2_c1_breathing.jl` | projected + breathing + GCL-ω + sub-stepped C≡1 | max\|C−1\| = **4.75e-14** (machine zero); receptor exactly 1.0 |
| `stage3_salinity.jl` | advect MARS3D `SAL` rigid-lid vs corrected, compare to truth | corrected RMS **~1–2 PSU** vs rigid-lid ~11–72 PSU (10–40× better) |

**Key implementation notes carried into production (from the scripts):** the Poisson needs a deep-mouth
(h>30 m) Dirichlet boundary + flood-fill to the mouth-connected component (else singular — the estuary
open boundary is not at the array edge); the PoC's explicit upwind needed sub-stepping (Courant<0.5, up to
~145 sub-steps/interval) — production replaces this with breathing-FFSL at Courant>1 (no sub-stepping).
These scripts are the small-grid oracle; production must reproduce them through the real engine path.
