# Stage 3: physical validation vs MARS3D's own salinity SAL (full 2012 file = ground truth).
# Advect C=SAL(t0) offline over ~1.5 tidal cycles with (a) RIGID-LID (static vol, distribute-Dtot ω, raw
# fluxes = current engine) and (b) CORRECTED (Poisson-projected fluxes + breathing vol + GCL ω). Open
# boundaries nudged to SAL. Compare each to SAL(t): does CORRECTED track the truth better?
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf, Statistics, SparseArrays, LinearAlgebra, Dates
const path=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2012.nc"
const RI,RJ=157,81; println("nthreads=",Threads.nthreads()); flush(stdout)

grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ds=NCDataset(path); state=initialize_state(grid, ds,(:C,))
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
traw=ds["time"][:]
tv = eltype(traw)<:Union{DateTime,Date} ? [Dates.value(DateTime(t)-DateTime(traw[1]))/1000.0 for t in traw] : Float64.(traw)
sig=Float64.(coalesce.(ds["SIG"][:],0.0)); wsig=Vector{Float64}(undef,nz+1);wsig[1]=-1;wsig[nz+1]=0
for k in 2:nz; wsig[k]=0.5*(sig[k-1]+sig[k]); end
dsig=[wsig[k+1]-wsig[k] for k in 1:nz]
readSAL(it)=Float64.(coalesce.(ds["SAL"][:,:,:,it],35.0))    # (nx,ny,nz)

wet(i,j)= 1<=i<=nx && 1<=j<=ny && grid.mask_rho[i+ng,j+ng] && grid.h[i+ng,j+ng]>1.0
hasnonwet(i,j)= !(wet(i-1,j)&&wet(i+1,j)&&wet(i,j-1)&&wet(i,j+1))
isopen(i,j)= wet(i,j) && hasnonwet(i,j) && grid.h[i+ng,j+ng]>30.0
dxo(i,j)=1/grid.pm[i+ng,j+ng]; dyo(i,j)=1/grid.pn[i+ng,j+ng]
const Dmin=0.1
function flood(); reach=falses(nx,ny); st=Tuple{Int,Int}[]
    for j in 1:ny,i in 1:nx; isopen(i,j)&&(reach[i,j]=true;push!(st,(i,j))); end
    while !isempty(st); (i,j)=pop!(st); for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)); (wet(ni,nj)&&!reach[ni,nj])&&(reach[ni,nj]=true;push!(st,(ni,nj))); end; end; reach; end
const reach=flood()
active(i,j)= wet(i,j) && reach[i,j] && !isopen(i,j)
function number_cells(); id=zeros(Int,nx,ny); N=0; for j in 1:ny,i in 1:nx; active(i,j)&&(N+=1;id[i,j]=N); end; id,N; end
const id, Nun = number_cells()
@printf("grid %d×%d×%d  unknowns=%d open=%d\n", nx,ny,nz,Nun,count(isopen(i,j) for i in 1:nx,j in 1:ny)); flush(stdout)

Vcell(i,j,k,H)=dxo(i,j)*dyo(i,j)*dsig[k]*H
facew(i,j,ni,nj,D)=begin Df=min(D(i,j),D(ni,nj)); if ni!=i; Lf=0.5*(dyo(i,j)+dyo(ni,nj));dc=0.5*(dxo(i,j)+dxo(ni,nj)); else; Lf=0.5*(dxo(i,j)+dxo(ni,nj));dc=0.5*(dyo(i,j)+dyo(ni,nj)); end; Lf*Df/dc end

# Build corrected (projected+breathing) fluxes/ω/depths into Ux,Uy,Wk,Hn,Hnp. Returns nothing.
function build_corrected!(Ux,Uy,Wk,Hn,Hnp,XEn,XEnp1,dt_h)
    u=state.u; v=state.v; XEmid=0.5.*(XEn.+XEnp1)
    D(i,j)= wet(i,j) ? max(grid.h[i+ng,j+ng]+XEmid[i,j],Dmin) : 0.0
    for j in 1:ny,i in 1:nx
        Hn[i,j]= wet(i,j) ? max(grid.h[i+ng,j+ng]+XEn[i,j],Dmin) : 0.0
        Hnp[i,j]=wet(i,j) ? max(grid.h[i+ng,j+ng]+XEnp1[i,j],Dmin) : 0.0
    end
    Draw=zeros(nx,ny); T=zeros(nx,ny)
    for j in 1:ny,i in 1:nx
        wet(i,j)||continue; ig,jg=i+ng,j+ng; dx=dxo(i,j);dy=dyo(i,j)
        DxE=min(D(i,j),D(i+1,j));DxW=min(D(i-1,j),D(i,j));DyN=min(D(i,j),D(i,j+1));DyS=min(D(i,j-1),D(i,j))
        s=0.0; for k in 1:nz; s+=u[ig+1,jg,k]*dy*dsig[k]*DxE-u[ig,jg,k]*dy*dsig[k]*DxW+v[ig,jg+1,k]*dx*dsig[k]*DyN-v[ig,jg,k]*dx*dsig[k]*DyS; end
        Draw[i,j]=s; T[i,j]=-dx*dy*(XEnp1[i,j]-XEn[i,j])/dt_h
    end
    II=Int[];JJ=Int[];VV=Float64[]; b=zeros(Nun)
    for j in 1:ny,i in 1:nx; active(i,j)||continue; p=id[i,j]; diag=0.0
        for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            if active(ni,nj); w=facew(i,j,ni,nj,D);diag+=w;push!(II,p);push!(JJ,id[ni,nj]);push!(VV,-w)
            elseif isopen(ni,nj); diag+=facew(i,j,ni,nj,D); end; end
        push!(II,p);push!(JJ,p);push!(VV,diag); b[p]=T[i,j]-Draw[i,j]; end
    φv=sparse(II,JJ,VV,Nun,Nun)\b
    φ=zeros(nx,ny); for j in 1:ny,i in 1:nx; active(i,j)&&(φ[i,j]=φv[id[i,j]]); end
    fill!(Ux,0.0);fill!(Uy,0.0)
    for j in 1:ny,i in 1:nx
        if wet(i-1,j)&&wet(i,j); ig,jg=i+ng,j+ng; dy=0.5*(dyo(i-1,j)+dyo(i,j)); Df=min(D(i-1,j),D(i,j))
            q=(active(i,j)||isopen(i,j))&&(active(i-1,j)||isopen(i-1,j)) ? facew(i,j,i-1,j,D)*(φ[i-1,j]-φ[i,j]) : 0.0
            for k in 1:nz; Ux[i,j,k]=u[ig,jg,k]*dy*dsig[k]*Df+q*dsig[k]; end; end
        if wet(i,j-1)&&wet(i,j); ig,jg=i+ng,j+ng; dx=0.5*(dxo(i,j-1)+dxo(i,j)); Df=min(D(i,j-1),D(i,j))
            q=(active(i,j)||isopen(i,j))&&(active(i,j-1)||isopen(i,j-1)) ? facew(i,j,i,j-1,D)*(φ[i,j-1]-φ[i,j]) : 0.0
            for k in 1:nz; Uy[i,j,k]=v[ig,jg,k]*dx*dsig[k]*Df+q*dsig[k]; end; end
    end
    fill!(Wk,0.0)
    for j in 2:ny-1,i in 2:nx-1; (active(i,j)||isopen(i,j))||continue; area=dxo(i,j)*dyo(i,j)
        for k in 1:nz; hdiv=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]
            dVk=area*dsig[k]*(Hnp[i,j]-Hn[i,j])/dt_h; Wk[i,j,k+1]=Wk[i,j,k]-(dVk+hdiv); end; end
    nothing
end

# Build rigid-lid (static vol, raw fluxes, distribute-Dtot ω) into Ux,Uy,Wk,Hn,Hnp(=H0 static).
function build_rigid!(Ux,Uy,Wk,Hn,Hnp,dt_h)
    u=state.u; v=state.v
    D(i,j)= wet(i,j) ? grid.h[i+ng,j+ng] : 0.0           # STATIC bathymetry
    for j in 1:ny,i in 1:nx; Hn[i,j]=D(i,j); Hnp[i,j]=D(i,j); end
    fill!(Ux,0.0);fill!(Uy,0.0)
    for j in 1:ny,i in 1:nx
        if wet(i-1,j)&&wet(i,j); ig,jg=i+ng,j+ng; dy=0.5*(dyo(i-1,j)+dyo(i,j)); Df=min(D(i-1,j),D(i,j))
            for k in 1:nz; Ux[i,j,k]=u[ig,jg,k]*dy*dsig[k]*Df; end; end
        if wet(i,j-1)&&wet(i,j); ig,jg=i+ng,j+ng; dx=0.5*(dxo(i,j-1)+dxo(i,j)); Df=min(D(i,j-1),D(i,j))
            for k in 1:nz; Uy[i,j,k]=v[ig,jg,k]*dx*dsig[k]*Df; end; end
    end
    fill!(Wk,0.0)                                          # ω: distribute Dtot (W_surf forced 0)
    for j in 2:ny-1,i in 2:nx-1; (active(i,j)||isopen(i,j))||continue
        Dtot=0.0; for k in 1:nz; Dtot+=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]; end
        for k in 1:nz; hdiv=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]
            Wk[i,j,k+1]=Wk[i,j,k]-hdiv+dsig[k]*Dtot; end; end   # ΣΔσ=1 ⇒ W_surf=0
    nothing
end

function substep!(Cb,Ca,Ux,Uy,Wk,Hn,Hnp,f0,f1,dt_s)
    for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue
        H0f=Hn[i,j]+f0*(Hnp[i,j]-Hn[i,j]); H1f=Hn[i,j]+f1*(Hnp[i,j]-Hn[i,j])
        for k in 1:nz
            hflux=Ux[i+1,j,k]*(Ux[i+1,j,k]>=0 ? Ca[i,j,k] : Ca[i+1,j,k]) - Ux[i,j,k]*(Ux[i,j,k]>=0 ? Ca[i-1,j,k] : Ca[i,j,k]) +
                  Uy[i,j+1,k]*(Uy[i,j+1,k]>=0 ? Ca[i,j,k] : Ca[i,j+1,k]) - Uy[i,j,k]*(Uy[i,j,k]>=0 ? Ca[i,j-1,k] : Ca[i,j,k])
            wt=Wk[i,j,k+1]; wb=Wk[i,j,k]
            vflux=(wt>=0 ? wt*Ca[i,j,k] : wt*Ca[i,j,min(k+1,nz)]) - (wb>=0 ? wb*Ca[i,j,max(k-1,1)] : wb*Ca[i,j,k])
            Cb[i,j,k]=(Vcell(i,j,k,H0f)*Ca[i,j,k]-dt_s*(hflux+vflux))/Vcell(i,j,k,H1f)
        end; end
    nothing
end
function courant(Ux,Uy,Wk,Hn,dt_h)
    m=0.0; for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; for k in 1:nz
        out=max(Ux[i+1,j,k],0)+max(-Ux[i,j,k],0)+max(Uy[i,j+1,k],0)+max(-Uy[i,j,k],0)+max(Wk[i,j,k+1],0)+max(-Wk[i,j,k],0)
        m=max(m,out*dt_h/max(Vcell(i,j,k,Hn[i,j]),1e-6)); end; end; m; end
setBC!(C,SAL)= for j in 1:ny,i in 1:nx; (isopen(i,j)||!active(i,j)) && wet(i,j) && (for k in 1:nz; C[i,j,k]=SAL[i,j,k]; end); end
rmserr(C,SAL)=begin s=0.0;n=0; for j in 2:ny-1,i in 2:nx-1; active(i,j)||continue; for k in 1:nz; s+=(C[i,j,k]-SAL[i,j,k])^2;n+=1; end; end; sqrt(s/n); end

function run()
    Uxc=zeros(nx,ny,nz);Uyc=zeros(nx,ny,nz);Wkc=zeros(nx,ny,nz+1);Hnc=zeros(nx,ny);Hnpc=zeros(nx,ny)
    Uxr=zeros(nx,ny,nz);Uyr=zeros(nx,ny,nz);Wkr=zeros(nx,ny,nz+1);Hnr=zeros(nx,ny);Hnpr=zeros(nx,ny)
    ip0=8000; T_M2=44712.0; nint=ceil(Int,1.5*T_M2/1800); dt_h=1800.0
    SAL0=readSAL(ip0)
    Cc=copy(SAL0); Cr=copy(SAL0); Ca=zeros(nx,ny,nz); Cb=zeros(nx,ny,nz)
    @printf("running %d intervals; init C=SAL. receptor SAL0=%.3f\n", nint, SAL0[RI,RJ,nz÷2]); flush(stdout)
    substepper!(C,Ux,Uy,Wk,Hn,Hnp,dt_h)=begin
        cour=courant(Ux,Uy,Wk,Hn,dt_h); M=max(1,ceil(Int,cour/0.5)); dt_s=dt_h/M
        Ca.=C; for s in 1:M; substep!(Cb,Ca,Ux,Uy,Wk,Hn,Hnp,(s-1)/M,s/M,dt_s); Ca,Cb=Cb,Ca; end; C.=Ca; M
    end
    for m in 1:nint
        ip=ip0+m-1; ip+1<=length(tv)||break
        XEn=Float64.(coalesce.(ds["XE"][:,:,ip],0.0)); XEnp1=Float64.(coalesce.(ds["XE"][:,:,ip+1],0.0))
        update_hydrodynamics!(state,grid,ds,hydro,0.5*(tv[ip]+tv[ip+1]); diagnose_w=false)
        SALn=readSAL(ip); SALnp=readSAL(ip+1)
        setBC!(Cc,SALn); setBC!(Cr,SALn)                  # nudge boundaries to truth at interval start
        build_corrected!(Uxc,Uyc,Wkc,Hnc,Hnpc,XEn,XEnp1,dt_h); Mc=substepper!(Cc,Uxc,Uyc,Wkc,Hnc,Hnpc,dt_h)
        build_rigid!(Uxr,Uyr,Wkr,Hnr,Hnpr,dt_h);            Mr=substepper!(Cr,Uxr,Uyr,Wkr,Hnr,Hnpr,dt_h)
        ec=rmserr(Cc,SALnp); er=rmserr(Cr,SALnp)
        (m%4==0||m==nint)&&(@printf("  m=%2d  RMSvsSAL: corrected=%.4f  rigid=%.4f   receptor(true/corr/rigid)=%.3f/%.3f/%.3f\n",
            m, ec, er, SALnp[RI,RJ,nz÷2], Cc[RI,RJ,nz÷2], Cr[RI,RJ,nz÷2]); flush(stdout))
    end
    nothing
end
run(); close(ds)
