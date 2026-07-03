# Stage 2 (v2, sub-stepped): projected + breathing + GCL-ω scheme, does it preserve C≡1 on the real field?
# Per 30-min hydro interval: Poisson-project the barotropic transport (div→T), layerise, breathe volumes,
# diagnose ω bottom-up from the SAME corrected fluxes, then SUB-STEP the tracer at Courant<1 (transport &
# ω held piecewise-constant, volume linear in t). Rigid-lid gave 0.52 dex rectified / factor-10 at receptor.
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf, Statistics, SparseArrays, LinearAlgebra
const path=raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"
const RI,RJ=157,81; println("nthreads=",Threads.nthreads()); flush(stdout)

grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ds=NCDataset(path); state=initialize_state(grid, ds,(:C,))
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
tv=Float64.(ds["time"][:])
sig=Float64.(coalesce.(ds["SIG"][:],0.0)); wsig=Vector{Float64}(undef,nz+1);wsig[1]=-1;wsig[nz+1]=0
for k in 2:nz; wsig[k]=0.5*(sig[k-1]+sig[k]); end
dsig=[wsig[k+1]-wsig[k] for k in 1:nz]

wet(i,j)= 1<=i<=nx && 1<=j<=ny && grid.mask_rho[i+ng,j+ng] && grid.h[i+ng,j+ng]>1.0
hasnonwet(i,j)= !(wet(i-1,j)&&wet(i+1,j)&&wet(i,j-1)&&wet(i,j+1))
isopen(i,j)= wet(i,j) && hasnonwet(i,j) && grid.h[i+ng,j+ng]>30.0
dxo(i,j)=1/grid.pm[i+ng,j+ng]; dyo(i,j)=1/grid.pn[i+ng,j+ng]
const Dmin=0.1
function flood()
    reach=falses(nx,ny); st=Tuple{Int,Int}[]
    for j in 1:ny,i in 1:nx; isopen(i,j)&&(reach[i,j]=true;push!(st,(i,j))); end
    while !isempty(st); (i,j)=pop!(st); for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)); (wet(ni,nj)&&!reach[ni,nj])&&(reach[ni,nj]=true;push!(st,(ni,nj))); end; end
    reach
end
const reach=flood()
active(i,j)= wet(i,j) && reach[i,j] && !isopen(i,j)
function number_cells(); id=zeros(Int,nx,ny); N=0; for j in 1:ny,i in 1:nx; active(i,j)&&(N+=1;id[i,j]=N); end; id,N; end
const id, Nun = number_cells()
@printf("unknowns=%d  open=%d\n", Nun, count(isopen(i,j) for i in 1:nx,j in 1:ny)); flush(stdout)

# preallocated per-interval fields
Ux=zeros(nx,ny,nz); Uy=zeros(nx,ny,nz); Wk=zeros(nx,ny,nz+1)   # corrected layer transports + ω volume flux
Hn=zeros(nx,ny); Hnp=zeros(nx,ny)                              # total depths at n, n+1 (floored)

function build_interval!(XEn,XEnp1,dt_h)
    u=state.u; v=state.v; XEmid=0.5.*(XEn.+XEnp1)
    D(i,j)= wet(i,j) ? max(grid.h[i+ng,j+ng]+XEmid[i,j],Dmin) : 0.0
    for j in 1:ny,i in 1:nx
        Hn[i,j] = wet(i,j) ? max(grid.h[i+ng,j+ng]+XEn[i,j],Dmin) : 0.0
        Hnp[i,j]= wet(i,j) ? max(grid.h[i+ng,j+ng]+XEnp1[i,j],Dmin) : 0.0
    end
    # raw barotropic divergence + target, then Poisson project
    Draw=zeros(nx,ny); T=zeros(nx,ny)
    for j in 1:ny,i in 1:nx
        wet(i,j)||continue; ig,jg=i+ng,j+ng; dx=dxo(i,j); dy=dyo(i,j)
        DxE=min(D(i,j),D(i+1,j));DxW=min(D(i-1,j),D(i,j));DyN=min(D(i,j),D(i,j+1));DyS=min(D(i,j-1),D(i,j))
        s=0.0; for k in 1:nz; s+=u[ig+1,jg,k]*dy*dsig[k]*DxE-u[ig,jg,k]*dy*dsig[k]*DxW+v[ig,jg+1,k]*dx*dsig[k]*DyN-v[ig,jg,k]*dx*dsig[k]*DyS; end
        Draw[i,j]=s; T[i,j]=-dx*dy*(XEnp1[i,j]-XEn[i,j])/dt_h
    end
    facew(i,j,ni,nj)=begin Df=min(D(i,j),D(ni,nj)); if ni!=i; Lf=0.5*(dyo(i,j)+dyo(ni,nj));dc=0.5*(dxo(i,j)+dxo(ni,nj)); else; Lf=0.5*(dxo(i,j)+dxo(ni,nj));dc=0.5*(dyo(i,j)+dyo(ni,nj)); end; Lf*Df/dc end
    II=Int[];JJ=Int[];VV=Float64[]; b=zeros(Nun)
    for j in 1:ny,i in 1:nx
        active(i,j)||continue; p=id[i,j]; diag=0.0
        for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            if active(ni,nj); w=facew(i,j,ni,nj);diag+=w;push!(II,p);push!(JJ,id[ni,nj]);push!(VV,-w)
            elseif isopen(ni,nj); diag+=facew(i,j,ni,nj); end
        end
        push!(II,p);push!(JJ,p);push!(VV,diag); b[p]=T[i,j]-Draw[i,j]
    end
    φv=sparse(II,JJ,VV,Nun,Nun)\b
    φ=zeros(nx,ny); for j in 1:ny,i in 1:nx; active(i,j)&&(φ[i,j]=φv[id[i,j]]); end
    # corrected layer transports at west (Ux) / south (Uy) faces
    fill!(Ux,0.0); fill!(Uy,0.0)
    for j in 1:ny,i in 1:nx
        if wet(i-1,j)&&wet(i,j)
            ig,jg=i+ng,j+ng; dy=0.5*(dyo(i-1,j)+dyo(i,j)); Df=min(D(i-1,j),D(i,j))
            q=(active(i,j)||isopen(i,j))&&(active(i-1,j)||isopen(i-1,j)) ? facew(i,j,i-1,j)*(φ[i-1,j]-φ[i,j]) : 0.0
            for k in 1:nz; Ux[i,j,k]=u[ig,jg,k]*dy*dsig[k]*Df + q*dsig[k]; end
        end
        if wet(i,j-1)&&wet(i,j)
            ig,jg=i+ng,j+ng; dx=0.5*(dxo(i,j-1)+dxo(i,j)); Df=min(D(i,j-1),D(i,j))
            q=(active(i,j)||isopen(i,j))&&(active(i,j-1)||isopen(i,j-1)) ? facew(i,j,i,j-1)*(φ[i,j-1]-φ[i,j]) : 0.0
            for k in 1:nz; Uy[i,j,k]=v[ig,jg,k]*dx*dsig[k]*Df + q*dsig[k]; end
        end
    end
    # GCL ω bottom-up per column
    fill!(Wk,0.0)
    for j in 2:ny-1,i in 2:nx-1
        (active(i,j)||isopen(i,j))||continue; area=dxo(i,j)*dyo(i,j)
        for k in 1:nz
            hdiv=Ux[i+1,j,k]-Ux[i,j,k]+Uy[i,j+1,k]-Uy[i,j,k]
            dVk=area*dsig[k]*(Hnp[i,j]-Hn[i,j])/dt_h
            Wk[i,j,k+1]=Wk[i,j,k]-(dVk+hdiv)
        end
    end
    return nothing
end

Vcell(i,j,k,H)=dxo(i,j)*dyo(i,j)*dsig[k]*H
# CFL: max Courant of the corrected fluxes over the interval
function max_courant(dt_h)
    m=0.0
    for j in 2:ny-1,i in 2:nx-1
        active(i,j)||continue
        for k in 1:nz
            out=max(Ux[i+1,j,k],0)+max(-Ux[i,j,k],0)+max(Uy[i,j+1,k],0)+max(-Uy[i,j,k],0)+max(Wk[i,j,k+1],0)+max(-Wk[i,j,k],0)
            m=max(m, out*dt_h/max(Vcell(i,j,k,Hn[i,j]),1e-6))
        end
    end
    m
end
# one sub-step: Cb <- update(Ca) with volumes at fractions f0->f1 of the interval
function substep!(Cb,Ca,f0,f1,dt_s)
    for j in 2:ny-1,i in 2:nx-1
        active(i,j)||continue
        Hn0=Hn[i,j]+f0*(Hnp[i,j]-Hn[i,j]); Hn1=Hn[i,j]+f1*(Hnp[i,j]-Hn[i,j])
        for k in 1:nz
            hflux=Ux[i+1,j,k]*(Ux[i+1,j,k]>=0 ? Ca[i,j,k] : Ca[i+1,j,k]) - Ux[i,j,k]*(Ux[i,j,k]>=0 ? Ca[i-1,j,k] : Ca[i,j,k]) +
                  Uy[i,j+1,k]*(Uy[i,j+1,k]>=0 ? Ca[i,j,k] : Ca[i,j+1,k]) - Uy[i,j,k]*(Uy[i,j,k]>=0 ? Ca[i,j-1,k] : Ca[i,j,k])
            wt=Wk[i,j,k+1]; wb=Wk[i,j,k]
            vflux=(wt>=0 ? wt*Ca[i,j,k] : wt*Ca[i,j,min(k+1,nz)]) - (wb>=0 ? wb*Ca[i,j,max(k-1,1)] : wb*Ca[i,j,k])
            Cb[i,j,k]=(Vcell(i,j,k,Hn0)*Ca[i,j,k]-dt_s*(hflux+vflux))/Vcell(i,j,k,Hn1)
        end
    end
    nothing
end

function run_all()
    C=ones(nx,ny,nz); Ca=ones(nx,ny,nz); Cb=ones(nx,ny,nz)
    ip0=length(tv)÷2; T_M2=44712.0; nint=ceil(Int,2*T_M2/1800); dt_h=1800.0
    @printf("running %d intervals (~%.1f M2 cycles)\n", nint, nint*1800/T_M2); flush(stdout)
    recC=Float64[]; peak=0.0; maxM=0
    for m in 1:nint
        ip=ip0+m-1; ip+1<=length(tv)||break
        XEn=coalesce.(ds["XE"][:,:,ip],0.0); XEnp1=coalesce.(ds["XE"][:,:,ip+1],0.0)
        update_hydrodynamics!(state,grid,ds,hydro,0.5*(tv[ip]+tv[ip+1]); diagnose_w=false)
        build_interval!(XEn,XEnp1,dt_h)
        cour=max_courant(dt_h); M=max(1,ceil(Int,cour/0.5)); dt_s=dt_h/M; maxM=max(maxM,M)
        Ca .= C
        for s in 1:M
            substep!(Cb,Ca, (s-1)/M, s/M, dt_s); Ca,Cb = Cb,Ca
        end
        C .= Ca
        mx=0.0; for j in 2:ny-1,i in 2:nx-1,k in 1:nz; active(i,j)&&(mx=max(mx,abs(C[i,j,k]-1.0))); end
        peak=max(peak,mx); push!(recC,C[RI,RJ,nz÷2])
        (m%8==0)&&(@printf("  interval %d/%d  M=%d  max|C-1|=%.3e  receptorC=%.8f\n",m,nint,M,mx,C[RI,RJ,nz÷2]); flush(stdout))
    end
    peak, recC, maxM
end
peak, recC, maxM = run_all()
close(ds)
@printf("\nSTAGE 2 RESULT: peak max|C-1| over run = %.3e  (=%.5f dex)  [max sub-steps/interval=%d]\n", peak, log10(1+peak), maxM)
@printf("receptor C range: [%.8f, %.8f]  (rigid-lid was 0.03..6.06)\n", minimum(recC),maximum(recC))
println(peak<1e-4 ? "PASS: C≡1 preserved — the 0.52 dex rigid-lid artifact is GONE." : "CHECK: residual not tiny.")
