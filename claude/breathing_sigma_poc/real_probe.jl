# C≡1 consistency probe on the REAL MARS3D CurviLoire velocity field (2010 subset). Same instrument
# as the fixture, now on true uz/vz over ~2.5 M2 cycles. The TRUE model keeps a uniform tracer ≡1
# exactly, so max|C-1| IS the rigid-lid (frozen-volume) transport error. Metrics: single-step
# continuity residual; interior peak (wet, h>1 m); and peak + tidally-rectified error at the intertidal
# receptor VILLES_MARTIN (ij=157,81, h=3.02 m, η/H≈0.9). Diffusion Kh=1.0 (production-like).
import Pkg
const HT = raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using NCDatasets, Printf, Statistics

const path = raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"
const RI, RJ = 157, 81           # receptor VILLES_MARTIN physical ij
const T = 44712.0                # M2 period [s]

println("nthreads = ", Threads.nthreads()); flush(stdout)
grid  = initialize_curvilinear_grid(path)
hydro = create_hydrodynamic_data_from_file(path)
ds    = NCDataset(path)
state = initialize_state(grid, ds, (:C,))
ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
tv = Float64.(ds["time"][:]); t0 = tv[1]

# Fixed FFSL timestep. FFSL is conservative + positivity-preserving -> stable at advective Courant>1,
# so we do NOT let the one degenerate dx=1 m cell collapse dt (that gave 0.2 s / ~5e5 steps). 60 s is a
# typical offline sub-step for 30-min MARS3D output; median-cell Courant stays modest.
update_hydrodynamics!(state, grid, ds, hydro, t0 + 5*T; diagnose_w=true)
umax = maximum(abs, state.u); vmax = maximum(abs, state.v)
dxs = [1.0/grid.pm[i+ng,j+ng] for i in 1:nx, j in 1:ny if grid.mask_rho[i+ng,j+ng] && grid.h[i+ng,j+ng]>1.0]
dt = 60.0
@printf("umax=%.2f vmax=%.2f m/s  dx median=%.0f p10=%.0f min(h>1m)=%.1f m  dt=%.0f s (median Courant≈%.2f)\n",
        umax, vmax, median(dxs), quantile(dxs,0.10), minimum(dxs), dt, umax*dt/median(dxs)); flush(stdout)

# single-step continuity residual on the real field (interior all-wet-neighbour cells)
function residual(state, grid)
    ng=grid.ng; u=state.u; v=state.v; w=state.w; fax=grid.face_area_x; fay=grid.face_area_y
    mr=0.0; mh=0.0; sc=0.0
    for k in 1:grid.nz, j in 2:grid.ny-1, i in 2:grid.nx-1
        ig,jg=i+ng,j+ng
        (grid.mask_rho[ig,jg]&&grid.mask_rho[ig-1,jg]&&grid.mask_rho[ig+1,jg]&&grid.mask_rho[ig,jg-1]&&grid.mask_rho[ig,jg+1]) || continue
        area=1.0/(grid.pm[ig,jg]*grid.pn[ig,jg])
        hd=u[ig+1,jg,k]*fax[ig+1,jg,k]-u[ig,jg,k]*fax[ig,jg,k]+v[ig,jg+1,k]*fay[ig,jg+1,k]-v[ig,jg,k]*fay[ig,jg,k]
        vd=(w[ig,jg,k+1]-w[ig,jg,k])*area
        mr=max(mr,abs(hd+vd)); mh=max(mh,abs(hd)); sc=max(sc,abs(u[ig,jg,k]*fax[ig,jg,k]))
    end
    sc=sc>0 ? sc : 1.0; (mr/sc, mh/sc)
end
rd, rz = residual(state, grid)
@printf("single-step residual: diagnosed=%.3e  zero_w=%.3e  (fraction of Dtot left uncompensated=%.2f)\n",
        rd, rz, rd/rz); flush(stdout)

# C≡1 run
fill!(state.tracers[:C], 1.0); C = state.tracers[:C]
wetH = [grid.mask_rho[i+ng,j+ng] && grid.h[i+ng,j+ng] > 1.0 for i in 1:nx, j in 1:ny]
nper = 2.0; tend = t0 + nper*T; nsteps = ceil(Int,(tend-t0)/dt)
@printf("run: %d steps over %.1f cycles\n", nsteps, nper); flush(stdout)
function interior_peak(C, grid, wetH)
    ng=grid.ng; m=0.0
    for k in 1:grid.nz, j in 4:grid.ny-3, i in 4:grid.nx-3
        wetH[i,j] || continue; d=abs(C[i+ng,j+ng,k]-1.0); d>m && (m=d)
    end; m
end
recpeak=0.0; peakall=0.0
acc=zeros(nz); nacc=0                          # receptor-column time-mean over last period
@printf("\n  %6s | %12s | %12s | %10s\n","t/T","peak(h>1m)","recept peak","recept Cmid")
for step in 1:nsteps
    t=t0+step*dt
    update_hydrodynamics!(state, grid, ds, hydro, t; diagnose_w=true)
    horizontal_transport!(state, grid, dt, :FFSL, 0.0, BoundaryCondition[]; Kh=1.0)
    vertical_transport!(state, grid, dt; Kz=1e-4)
    rc = maximum(abs, @view C[RI+ng, RJ+ng, :]) - 0.0
    rcp = maximum(abs.(@view(C[RI+ng, RJ+ng, :]) .- 1.0))
    global recpeak = max(recpeak, rcp)
    if (t - t0) >= (nper-1)*T
        for k in 1:nz; acc[k]+=C[RI+ng,RJ+ng,k]; end; global nacc+=1
    end
    if step % max(1,nsteps÷30)==0 || step==nsteps
        pa=interior_peak(C,grid,wetH); global peakall=max(peakall,pa)
        @printf("  %6.3f | %12.4e | %12.4e | %10.5f\n", (t-t0)/T, pa, rcp, C[RI+ng,RJ+ng,nz÷2]); flush(stdout)
    end
end
close(ds)
rect = nacc>0 ? maximum(abs.(acc./nacc .- 1.0)) : NaN
@printf("\nREAL-FIELD RESULT (receptor η/H≈0.9):\n")
@printf("  interior peak |C-1| (wet, h>1m)      = %.4e  (%.3f log10)\n", peakall, log10(1+peakall))
@printf("  receptor peak |C-1|                  = %.4e  (%.3f log10)\n", recpeak, log10(1+recpeak))
@printf("  receptor tidally-RECTIFIED |mean-1|  = %.4e  (%.3f log10)\n", rect, log10(1+rect))
@printf("  (τ per-winter loading ≈ 0.524 log10 for reference)\n")
