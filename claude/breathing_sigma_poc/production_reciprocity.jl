# Reverse-time breathing ADJOINT DUALITY (robust, array-based): ⟨M a, b⟩_V = ⟨a, M* b⟩_V for broad
# random fields on a DEEP INTERIOR support (always-wet, far from open boundary + drying). Residual =
# FCT-limiter nonlinearity + small V-variation. Weight W = area·Δσ·H0.
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.ProjectionModule
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

Wt(i,j,k)=proj.dxo[i,j]*proj.dyo[i,j]*proj.dsig[k]*grid.h[i+ng,j+ng]
Random.seed!(7)
A=zeros(Float32,nx,ny,nz); B=zeros(Float32,nx,ny,nz)
for j in 1:ny,i in 1:nx; supp[i,j] || continue; for k in 1:nz; A[i,j,k]=rand(); B[i,j,k]=rand(); end; end

run1(seedfield; rev=false, org=0.0, ts=t0, te=t1) = begin
    ds=NCDataset(path); s=initialize_state(grid, ds,(:C,))
    for j in 1:ny,i in 1:nx,k in 1:nz; s.tracers[:C][i+ng,j+ng,k]=seedfield[i,j,k]; end
    f=run_simulation(grid, s, PointSource[], ts, te, 1800.0; ds=ds, hydro_data=hydro, breathing=true,
        breathing_camb=0.0, reverse_time=rev, reverse_time_origin=org,
        use_adaptive_dt=true, cfl_max=0.4, dt_max=1800.0, dt_min=0.01, Kz=1e-3, write_full_state=false)
    out=copy(f.tracers[:C]); close(ds); out
end
Ma=run1(A)                                    # forward M a
Mb=run1(B; rev=true, org=t1, ts=0.0, te=t1-t0) # reverse M* b
function dualprods()
    ip=0.0; iq=0.0
    for j in 1:ny,i in 1:nx; supp[i,j] || continue; for k in 1:nz
        ip += Wt(i,j,k)*Float64(Ma[i+ng,j+ng,k])*Float64(B[i,j,k])
        iq += Wt(i,j,k)*Float64(A[i,j,k])*Float64(Mb[i+ng,j+ng,k])
    end; end
    ip, iq
end
ip, iq = dualprods()
@printf("\n<Ma,b> = %.6e\n<a,M*b> = %.6e\nadjoint duality ratio = %.5f   rel err = %.2f%%\n",
        ip, iq, ip/iq, 100*abs(ip-iq)/max(abs(ip),abs(iq)))
println(abs(ip-iq)/max(abs(ip),abs(iq)) < 0.05 ?
        "RECIPROCITY PASS: reverse-time breathing is the adjoint to within the FCT floor (<5%)." :
        "RECIPROCITY: residual above 5% (FCT limiter + V-variation, or remaining replay).")
