# HEADLINE VALIDATION: does the PRODUCTION breathing code reproduce MARS3D's own salinity SAL?
# Replicates PoC stage3 methodology (init C=SAL(t0); each interval nudge open/inactive boundary cells to
# truth SAL; advect ~1.5 tidal cycles; RMS over active interior vs SAL(t)) but drives the CORRECTED case
# with the production ProjectionModule (project!) + BreathingTransportModule (breathing_transport!, Strang).
# Rigid-lid baseline = static volume + raw fluxes + distribute-Dtot ω (the current engine), explicit upwind.
# PoC result to reproduce: corrected RMS ~1-2 PSU vs rigid ~11-72 PSU.
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using HydrodynamicTransport.ProjectionModule
using HydrodynamicTransport.BreathingTransportModule
using NCDatasets, Printf, Statistics, Dates
const path=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2012.nc"
const RI,RJ=157,81; println("nthreads=",Threads.nthreads()); flush(stdout)

grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ds=NCDataset(path); state=initialize_state(grid, ds,(:C,))
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz; mx,my=nx+2ng,ny+2ng
traw=ds["time"][:]
tv = eltype(traw)<:Union{DateTime,Date} ? [Dates.value(DateTime(t)-DateTime(traw[1]))/1000.0 for t in traw] : Float64.(traw)
proj=build_projector(grid; h_open=30.0, wet_min=1.0, D_min=0.1); bw=build_breathing_work(grid)
dsig=proj.dsig; dxo(i,j)=proj.dxo[i,j]; dyo(i,j)=proj.dyo[i,j]
inb(i,j)= 1<=i<=nx && 1<=j<=ny
wet(i,j)=inb(i,j)&&proj.wet[i,j]; isopen(i,j)=inb(i,j)&&proj.isopen[i,j]; active(i,j)=inb(i,j)&&proj.active[i,j]
readSAL(it)=Float64.(coalesce.(ds["SAL"][:,:,:,it],35.0))
Vcell(i,j,k,H)=dxo(i,j)*dyo(i,j)*dsig[k]*H
@printf("grid %d×%d×%d  active=%d open=%d\n", nx,ny,nz, count(proj.active), count(proj.isopen)); flush(stdout)

# nudge open + inactive-wet cells to truth SAL (both schemes get identical boundaries)
function setBC_prod!(SAL)   # into state.tracers[:C] (padded)
    C=state.tracers[:C]
    @inbounds for j in 1:ny,i in 1:nx
        (wet(i,j) && (isopen(i,j) || !active(i,j))) || continue
        for k in 1:nz; C[i+ng,j+ng,k]=Float32(SAL[i,j,k]); end
    end
end
setBC_rig!(C,SAL)= @inbounds for j in 1:ny,i in 1:nx
    (wet(i,j)&&(isopen(i,j)||!active(i,j))) && (for k in 1:nz; C[i,j,k]=SAL[i,j,k]; end); end
rms_prod(SAL)=begin C=state.tracers[:C]; s=0.0;n=0
    @inbounds for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; for k in 1:nz; s+=(Float64(C[i+ng,j+ng,k])-SAL[i,j,k])^2;n+=1; end; end; sqrt(s/max(n,1)) end
rms_rig(C,SAL)=begin s=0.0;n=0; for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; for k in 1:nz; s+=(C[i,j,k]-SAL[i,j,k])^2;n+=1; end; end; sqrt(s/max(n,1)) end

# --- rigid-lid baseline (static vol, raw fluxes, distribute-Dtot ω, explicit upwind sub-stepped) ---
function build_rigid!(Ux,Uy,Wk)
    u=state.u; v=state.v; D(i,j)= wet(i,j) ? grid.h[i+ng,j+ng] : 0.0
    fill!(Ux,0.0);fill!(Uy,0.0)
    for j in 1:ny,i in 1:nx
        if wet(i-1,j)&&wet(i,j); ig,jg=i+ng,j+ng; dy=0.5*(dyo(i-1,j)+dyo(i,j)); Df=min(D(i-1,j),D(i,j)); for k in 1:nz; Ux[i,j,k]=u[ig,jg,k]*dy*dsig[k]*Df; end; end
        if wet(i,j-1)&&wet(i,j); ig,jg=i+ng,j+ng; dx=0.5*(dxo(i,j-1)+dxo(i,j)); Df=min(D(i,j-1),D(i,j)); for k in 1:nz; Uy[i,j,k]=v[ig,jg,k]*dx*dsig[k]*Df; end; end
    end
    fill!(Wk,0.0)
    for j in 2:ny-1,i in 2:nx-1; (active(i,j)||isopen(i,j))||continue
        Dtot=0.0; for k in 1:nz; Dtot+=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]; end
        for k in 1:nz; hdiv=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]; Wk[i,j,k+1]=Wk[i,j,k]-hdiv+dsig[k]*Dtot; end; end
end
function substep_rig!(Cb,Ca,Ux,Uy,Wk,dt_s)   # static H0
    for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; H=grid.h[i+ng,j+ng]
        for k in 1:nz
            hflux=Ux[i+1,j,k]*(Ux[i+1,j,k]>=0 ? Ca[i,j,k] : Ca[i+1,j,k]) - Ux[i,j,k]*(Ux[i,j,k]>=0 ? Ca[i-1,j,k] : Ca[i,j,k]) +
                  Uy[i,j+1,k]*(Uy[i,j+1,k]>=0 ? Ca[i,j,k] : Ca[i,j+1,k]) - Uy[i,j,k]*(Uy[i,j,k]>=0 ? Ca[i,j-1,k] : Ca[i,j,k])
            wt=Wk[i,j,k+1];wb=Wk[i,j,k]; vflux=(wt>=0 ? wt*Ca[i,j,k] : wt*Ca[i,j,min(k+1,nz)])-(wb>=0 ? wb*Ca[i,j,max(k-1,1)] : wb*Ca[i,j,k])
            Cb[i,j,k]=(Vcell(i,j,k,H)*Ca[i,j,k]-dt_s*(hflux+vflux))/Vcell(i,j,k,H)
        end; end
end
function cour_rig(Ux,Uy,Wk,dt_h)
    m=0.0; for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; for k in 1:nz
        out=max(Ux[i+1,j,k],0)+max(-Ux[i,j,k],0)+max(Uy[i,j+1,k],0)+max(-Uy[i,j,k],0)+max(Wk[i,j,k+1],0)+max(-Wk[i,j,k],0)
        m=max(m,out*dt_h/max(Vcell(i,j,k,grid.h[i+ng,j+ng]),1e-6)); end; end; m end

Uxr=zeros(nx,ny,nz);Uyr=zeros(nx,ny,nz);Wkr=zeros(nx,ny,nz+1); Car=zeros(nx,ny,nz);Cbr=zeros(nx,ny,nz)
ip0=8000; T_M2=44712.0; dt_h=1800.0; nint=16   # ~8 h (~0.6 M2 cycle) — enough to show the corrected/rigid contrast
SAL0=readSAL(ip0)
for j in 1:ny,i in 1:nx,k in 1:nz; state.tracers[:C][i+ng,j+ng,k]=Float32(SAL0[i,j,k]); end   # corrected init
Cr=copy(SAL0)                                                                                  # rigid init
@printf("running %d intervals from ip=%d; receptor SAL0=%.3f\n", nint, ip0, SAL0[RI,RJ,nz÷2]); flush(stdout)

for m in 1:nint
    global Car, Cbr
    ip=ip0+m-1; ip+1<=length(tv)||break
    XEn=Float64.(coalesce.(ds["XE"][:,:,ip],0.0)); XEnp1=Float64.(coalesce.(ds["XE"][:,:,ip+1],0.0))
    update_hydrodynamics!(state,grid,ds,hydro,0.5*(tv[ip]+tv[ip+1]); diagnose_w=false)
    SALn=readSAL(ip); SALnp=readSAL(ip+1)
    # --- CORRECTED (production) ---
    setBC_prod!(SALn)
    project!(proj, grid, state.u, state.v, XEn, XEnp1, tv[ip+1]-tv[ip]); pad_transports!(bw, proj)
    Mc=max(1, ceil(Int, breathing_courant(proj,bw)*dt_h/0.5)); dts=dt_h/Mc
    for s in 1:Mc; breathing_transport!(state, proj, bw, grid, dts, (s-1)/Mc; camb=35.0, Kz=0.0); end
    # --- RIGID (baseline) ---
    setBC_rig!(Cr,SALn)
    build_rigid!(Uxr,Uyr,Wkr); Mr=max(1,ceil(Int,cour_rig(Uxr,Uyr,Wkr,dt_h)/0.5)); dtr=dt_h/Mr
    Car.=Cr; for s in 1:Mr; substep_rig!(Cbr,Car,Uxr,Uyr,Wkr,dtr); Car,Cbr=Cbr,Car; end; Cr.=Car
    ec=rms_prod(SALnp); er=rms_rig(Cr,SALnp)
    @printf("  m=%2d Mc=%d Mr=%d  RMSvsSAL: corrected=%.4f  rigid=%.4f   receptor(true/corr/rigid)=%.3f/%.3f/%.3f\n",
        m, Mc, Mr, ec, er, SALnp[RI,RJ,nz÷2], Float64(state.tracers[:C][RI+ng,RJ+ng,nz÷2]), Cr[RI,RJ,nz÷2]); flush(stdout)
end
close(ds)
println("DONE salinity validation.")
