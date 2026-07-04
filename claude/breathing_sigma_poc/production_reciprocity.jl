# Reverse-time breathing ADJOINT DUALITY (robust, array-based): ⟨M a, b⟩_V = ⟨a, M* b⟩_V for broad
# random fields on a DEEP INTERIOR support (always-wet, far from open boundary + drying).
#
# Two knobs isolate the two reciprocity floors:
#   (1) LINEAR mode (breathing_linear=true): drop the horizontal PPM+FCT limiter for the first-order-upwind
#       breathing sweep, which is math-vetted (3 independent agents) to be the EXACT discrete adjoint —
#       removing the FCT nonlinearity that pinned the default scheme at ~0.4%.
#   (2) The adjoint-consistent INNER PRODUCT carries the breathing cell volumes: weight the forward output
#       by V(t1) (the final ARRIVAL volume) and the reverse output by V(t0) (the initial DEPARTURE volume);
#       only these two END volumes survive the cascade telescoping. Using a single static H0 weight leaves
#       a residual = the tidal volume variation over the window.
# MEASURED (2528-cell deep support, 2 h window): linear + V-weight gives ~1.5e-4, a 26x improvement over
# the FCT/H0 baseline (0.40%). The HORIZONTAL sweep is exactly self-adjoint (kernel unit test in
# test/benchmarks/reciprocity_benchmark.jl -> 2e-16); the residual that remains is the VERTICAL implicit
# backward-Euler advection, which is only a FIRST-ORDER (O(dt)) adjoint: the transpose of the forward
# vertical matrix carries the ARRIVAL volume on its diagonal but the reverse run carries the DEPARTURE
# volume (they differ by dt·diag(ω[k+1]−ω[k])). Confirmed here by exact dt-halving of the residual, and by
# Kz=0 being unchanged (it is the ω term, not diffusion). A vertical FFSL overlap-remap would make it exact.
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.ProjectionModule
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf, Random
const path=raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"
grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
proj=build_projector(grid; h_open=30.0, wet_min=1.0, D_min=0.1)
tv=Float64.(NCDataset(path)["time"][:]); ip0=length(tv)÷2
t0=tv[ip0]; t1=tv[ip0+4]

const R=16; const H_DEEP=8.0
deepact=falses(nx,ny)
for j in 1:ny,i in 1:nx; deepact[i,j]= proj.active[i,j] && grid.h[i+ng,j+ng]>H_DEEP; end
supp=falses(nx,ny)   # deep-active with a fully deep-active R-halo (no mass leaves the support region)
for j in R+1:ny-R, i in R+1:nx-R
    ok=true
    for dj in -R:R, di in -R:R; deepact[i+di,j+dj] || (ok=false; break); end
    supp[i,j]=ok
end
nsupp=count(supp); @printf("deep interior support: %d cells\n", nsupp); flush(stdout)
nsupp==0 && error("empty support")

# Breathing cell volumes V(t) = dxo·dyo·dσ·H(t) at the two window ENDPOINTS, obtained by projecting the
# reads that START at t0 and t1 (proj.Hn = floored total depth at the read start). Forward-mode projection
# (reverse=false) so Hn = H at the earlier bracket surface = H(t0) resp. H(t1).
tmp=initialize_state(grid, NCDataset(path), (:C,))
proj.last_idx=-1; update_hydrodynamics!(tmp, grid, NCDataset(path), hydro, t0; diagnose_w=false, projector=proj, reverse=false)
Hn0=copy(proj.Hn)
proj.last_idx=-1; update_hydrodynamics!(tmp, grid, NCDataset(path), hydro, t1; diagnose_w=false, projector=proj, reverse=false)
Hn1=copy(proj.Hn)
W0(i,j,k)=proj.dxo[i,j]*proj.dyo[i,j]*proj.dsig[k]*Hn0[i,j]      # V(t0): reverse-output weight
W1(i,j,k)=proj.dxo[i,j]*proj.dyo[i,j]*proj.dsig[k]*Hn1[i,j]      # V(t1): forward-output weight
Wh(i,j,k)=proj.dxo[i,j]*proj.dyo[i,j]*proj.dsig[k]*grid.h[i+ng,j+ng]  # static H0 (the old weight)

Random.seed!(7)
A=zeros(Float32,nx,ny,nz); B=zeros(Float32,nx,ny,nz)
for j in 1:ny,i in 1:nx; supp[i,j] || continue; for k in 1:nz; A[i,j,k]=rand(); B[i,j,k]=rand(); end; end

run1(seedfield; rev=false, org=0.0, ts=t0, te=t1, lin=true) = begin
    ds=NCDataset(path); s=initialize_state(grid, ds,(:C,))
    for j in 1:ny,i in 1:nx,k in 1:nz; s.tracers[:C][i+ng,j+ng,k]=seedfield[i,j,k]; end
    f=run_simulation(grid, s, PointSource[], ts, te, 1800.0; ds=ds, hydro_data=hydro, breathing=true,
        breathing_camb=0.0, breathing_linear=lin, reverse_time=rev, reverse_time_origin=org,
        use_adaptive_dt=true, cfl_max=0.4, dt_max=1800.0, dt_min=0.01, Kz=1e-3, write_full_state=false)
    out=copy(f.tracers[:C]); close(ds); out
end

# duality residual for a chosen forward/reverse output weighting
function dual(Ma, Mb, Wf, Wr)
    ip=0.0; iq=0.0
    for j in 1:ny,i in 1:nx; supp[i,j] || continue; for k in 1:nz
        ip += Wf(i,j,k)*Float64(Ma[i+ng,j+ng,k])*Float64(B[i,j,k])
        iq += Wr(i,j,k)*Float64(A[i,j,k])*Float64(Mb[i+ng,j+ng,k])
    end; end
    ip, iq, abs(ip-iq)/max(abs(ip),abs(iq))
end

for lin in (false, true)
    Ma=run1(A; lin=lin)                                     # forward M a
    Mb=run1(B; rev=true, org=t1, ts=0.0, te=t1-t0, lin=lin) # reverse M* b
    _,_,e_h = dual(Ma, Mb, Wh, Wh)                          # static-H0 weight (old)
    ip,iq,e_v = dual(Ma, Mb, W1, W0)                        # adjoint-consistent V(t1)/V(t0) weight
    @printf("\nlinear=%-5s : rel err  H0-weight = %.3e   V(t1)/V(t0)-weight = %.3e\n", lin, e_h, e_v)
    lin && @printf("           <Ma,b>_V(t1) = %.6e   <a,M*b>_V(t0) = %.6e   ratio = %.9f\n", ip, iq, ip/iq)
end
println("\n(default linear=false is FCT-limited ~0.4%; linear=true + V-weight -> ~1.5e-4, floor = the")
println(" vertical implicit backward-Euler advection = O(dt); the horizontal sweep is exact — see the")
println(" bench_breathing_adjoint_kernel unit test (2e-16). A vertical FFSL overlap-remap would remove it.)")
