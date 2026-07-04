# Wet/dry parking validation on the real 2010 CurviLoire slab (opt-in `breathing_parking`).
# Confirms the agent-vetted claims + measures the benefit:
#   (1) C≡1 stays bit-exact with parking on.
#   (2) The forward and reverse parked masks are IDENTICAL (min(Hn,Hnp) is swap-invariant) — the
#       reverse-time reciprocity requirement.
#   (3) Parking preserves div(U*)=T on active cells AND removes an intermittent continuity BLOW-UP that
#       the rigid floor suffers at near-singular drying cells (parking OFF hit ~5e16 at a drying window).
#   (4) The intertidal Courant collapse is reduced — fewer adaptive sub-steps at low water.
# Fix the `path` constant, then: julia +release -t auto --project=<HydrodynamicTransport> this_file.
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.ProjectionModule, HydrodynamicTransport.BreathingTransportModule
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf
const path=raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"
const DPARK=0.5
grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
tv=Float64.(NCDataset(path)["time"][:]); nT=length(tv)
st=initialize_state(grid, NCDataset(path), (:C,))

# (2)+(3): project a read; report parked mask, forward-vs-reverse identity, and continuity residual off/on
function probe(k)
    poff=build_projector(grid; parking=false); pon=build_projector(grid; parking=true, D_park=DPARK)
    poff.last_idx=-1; update_hydrodynamics!(st,grid,NCDataset(path),hydro,tv[k];diagnose_w=false,projector=poff,reverse=false)
    pon.last_idx=-1;  update_hydrodynamics!(st,grid,NCDataset(path),hydro,tv[k];diagnose_w=false,projector=pon, reverse=false)
    pr=build_projector(grid; parking=true, D_park=DPARK)
    pr.last_idx=-1;   update_hydrodynamics!(st,grid,NCDataset(path),hydro,tv[k];diagnose_w=false,projector=pr, reverse=true)
    dry=count(pon.wet[i,j]&&min(pon.Hn[i,j],pon.Hnp[i,j])<DPARK for j in 1:ny,i in 1:nx)
    @printf("k=%d drying(<%.1fm)=%d  parked=%d  fwd==rev mask: %s  continuity max OFF=%.2e ON=%.2e\n",
            k, DPARK, dry, count(pon.parked), pon.parked==pr.parked, continuity_residual(poff).max, continuity_residual(pon).max)
end
probe(4); flush(stdout)

# (4): scan a tidal window for the max-drying read; report the sub-step count Mc = ΔT·Courant/cfl off vs on
function courant_scan()
    pon=build_projector(grid; parking=true, D_park=DPARK); bw=build_breathing_work(grid)
    poff=build_projector(grid; parking=false); bwo=build_breathing_work(grid)
    best_dry=-1; best=(k=0,coff=0.0,con=0.0)
    for k in 4:2:60
        k+1>nT && break
        pon.last_idx=-1; update_hydrodynamics!(st,grid,NCDataset(path),hydro,tv[k];diagnose_w=false,projector=pon,reverse=false)
        pad_transports!(bw,pon); con=breathing_courant(pon,bw)
        poff.last_idx=-1; update_hydrodynamics!(st,grid,NCDataset(path),hydro,tv[k];diagnose_w=false,projector=poff,reverse=false)
        pad_transports!(bwo,poff); coff=breathing_courant(poff,bwo)
        dry=count(pon.wet[i,j]&&min(pon.Hn[i,j],pon.Hnp[i,j])<DPARK for j in 1:ny,i in 1:nx)
        dry>best_dry && (best_dry=dry; best=(k=k,coff=coff,con=con))
    end
    best_dry, best
end
bd,b=courant_scan()
@printf("max-drying read k=%d (dry=%d): sub-steps/read Mc OFF=%.0f ON=%.0f  (%.2fx fewer with parking)\n",
        b.k, bd, 1800*b.coff/0.4, 1800*b.con/0.4, b.coff/max(b.con,1e-30))

# (1): C≡1 bit-exact through run_simulation over a read, parking off vs on
function c1(; parking)
    ds=NCDataset(path); s=initialize_state(grid, ds,(:C,)); fill!(s.tracers[:C], 1.0)
    f=run_simulation(grid, s, PointSource[], tv[4], tv[8], 1800.0; ds=ds, hydro_data=hydro, breathing=true,
        breathing_camb=1.0, breathing_parking=parking, use_adaptive_dt=true, cfl_max=0.4, dt_max=1800.0,
        dt_min=0.01, Kz=1e-3, write_full_state=false)
    ci=f.tracers[:C][ng+1:ng+nx, ng+1:ng+ny, :]; close(ds); maximum(abs.(ci .- 1.0))
end
@printf("C≡1 max|C-1| over a read: parking OFF=%.2e  ON=%.2e\n", c1(parking=false), c1(parking=true))
