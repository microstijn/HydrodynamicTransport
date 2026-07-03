# Stage 1 (make-or-break): depth-weighted barotropic Poisson projection on the real MARS3D 2010 slab.
# Solve  Σ_{n~i} w_f (φ_i - φ_n) = r_i = T_i - Draw_i ,  w_f = L_f·D_f/Δ_f  (∇·(D∇φ) form),
# land faces dropped (Neumann), wet DOMAIN-EDGE cells = open boundary Dirichlet φ=0.
# Corrected transport U_f = U_f^raw + w_f(φ_L - φ_R) makes div U = T exactly.
# PASS = corrected divergence → T to ~machine zero AND |u'|/|u| ≈ 0.1 (bounded, not ~1).
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf, Statistics, SparseArrays, LinearAlgebra
const path=raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"
const RI,RJ=157,81
println("nthreads=",Threads.nthreads()); flush(stdout)

grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ds=NCDataset(path); state=initialize_state(grid, ds,(:C,))
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
tv=Float64.(ds["time"][:])
sig=Float64.(coalesce.(ds["SIG"][:],0.0)); wsig=Vector{Float64}(undef,nz+1);wsig[1]=-1;wsig[nz+1]=0
for k in 2:nz; wsig[k]=0.5*(sig[k-1]+sig[k]); end
dsig=[wsig[k+1]-wsig[k] for k in 1:nz]
ip=length(tv)÷2; tn,tnp1=tv[ip],tv[ip+1]; dtf=tnp1-tn
XEn=coalesce.(ds["XE"][:,:,ip],0.0); XEnp1=coalesce.(ds["XE"][:,:,ip+1],0.0); XEmid=0.5.*(XEn.+XEnp1)
update_hydrodynamics!(state,grid,ds,hydro,0.5*(tn+tnp1); diagnose_w=false); u=state.u; v=state.v; close(ds)

wet(i,j)= 1<=i<=nx && 1<=j<=ny && grid.mask_rho[i+ng,j+ng] && grid.h[i+ng,j+ng]>1.0
hasnonwet(i,j)= !(wet(i-1,j)&&wet(i+1,j)&&wet(i,j-1)&&wet(i,j+1))
# Open boundary (Dirichlet φ=0) = the DEEP seaward mouth: wet-region-boundary cells with h>30 m.
# Shallow coastline boundary stays Neumann (land faces dropped). This is the physical prism exit.
isedge(i,j)= wet(i,j) && hasnonwet(i,j) && grid.h[i+ng,j+ng]>30.0
Dcell(i,j)= wet(i,j) ? max(grid.h[i+ng,j+ng]+XEmid[i,j],0.0) : 0.0
dxo(i,j)=1/grid.pm[i+ng,j+ng]; dyo(i,j)=1/grid.pn[i+ng,j+ng]

# per-cell raw barotropic divergence Draw and target T
Draw=zeros(nx,ny); T=zeros(nx,ny)
for j in 1:ny, i in 1:nx
    wet(i,j) || continue
    ig,jg=i+ng,j+ng; dx=dxo(i,j); dy=dyo(i,j)
    DxE=min(Dcell(i,j),Dcell(i+1,j)); DxW=min(Dcell(i-1,j),Dcell(i,j))
    DyN=min(Dcell(i,j),Dcell(i,j+1)); DyS=min(Dcell(i,j-1),Dcell(i,j))
    s=0.0
    for k in 1:nz
        s += u[ig+1,jg,k]*dy*dsig[k]*DxE - u[ig,jg,k]*dy*dsig[k]*DxW +
             v[ig,jg+1,k]*dx*dsig[k]*DyN - v[ig,jg,k]*dx*dsig[k]*DyS
    end
    Draw[i,j]=s; T[i,j]=-dx*dy*(XEnp1[i,j]-XEn[i,j])/dtf
end

# flood-fill the wet component connected to the open (mouth) boundary; isolated pools get no correction
function flood()
    reach=falses(nx,ny); st=Tuple{Int,Int}[]
    for j in 1:ny, i in 1:nx; isedge(i,j) && (reach[i,j]=true; push!(st,(i,j))); end
    while !isempty(st)
        (i,j)=pop!(st)
        for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            if wet(ni,nj) && !reach[ni,nj]; reach[ni,nj]=true; push!(st,(ni,nj)); end
        end
    end
    reach
end
reach=flood()
active(i,j)= wet(i,j) && reach[i,j] && !isedge(i,j)   # interior unknowns (mouth-connected)

# number interior wet cells (unknowns); edge wet cells are Dirichlet φ=0
function number_cells()
    id=zeros(Int,nx,ny); N=0
    for j in 1:ny, i in 1:nx
        active(i,j) && (N+=1; id[i,j]=N)
    end
    id, N
end
id, N = number_cells()
Nedge=count(wet(i,j) && isedge(i,j) for i in 1:nx, j in 1:ny)
Niso=count(wet(i,j) && !reach[i,j] for i in 1:nx, j in 1:ny)
@printf("isolated (unreachable) wet cells excluded: %d\n", Niso)
@printf("wet(h>1m)=%d  interior unknowns N=%d  open-boundary Dirichlet cells=%d\n",
        count(wet(i,j) for i in 1:nx,j in 1:ny), N, Nedge); flush(stdout)

# assemble  Σ w_f(φ_i-φ_n)=r_i
II=Int[]; JJ=Int[]; VV=Float64[]; b=zeros(N)
facew(i,j,ni,nj)= begin
    Df=min(Dcell(i,j),Dcell(ni,nj))
    if ni!=i;  dyo_f=0.5*(dyo(i,j)+dyo(ni,nj)); Lf=dyo_f; dcen=0.5*(dxo(i,j)+dxo(ni,nj))
    else;      dxo_f=0.5*(dxo(i,j)+dxo(ni,nj)); Lf=dxo_f; dcen=0.5*(dyo(i,j)+dyo(ni,nj)); end
    Lf*Df/dcen
end
for j in 1:ny, i in 1:nx
    active(i,j) || continue
    p=id[i,j]; diag=0.0
    for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
        if active(ni,nj)
            w=facew(i,j,ni,nj); diag+=w; push!(II,p);push!(JJ,id[ni,nj]);push!(VV,-w)
        elseif isedge(ni,nj)                       # Dirichlet φ=0 (mouth): diag only
            w=facew(i,j,ni,nj); diag+=w
        end                                        # else land/isolated: Neumann (drop)
    end
    push!(II,p);push!(JJ,p);push!(VV,diag); b[p]=T[i,j]-Draw[i,j]
end
A=sparse(II,JJ,VV,N,N)
@printf("assembled A: %d×%d, nnz=%d, symmetric=%s\n", N,N,nnz(A), issymmetric(A)); flush(stdout)
φv = A\b
φ=zeros(nx,ny); for j in 1:ny,i in 1:nx; active(i,j) && (φ[i,j]=φv[id[i,j]]); end
@printf("solve rel residual ||Aφ-b||/||b|| = %.3e\n", norm(A*φv-b)/norm(b)); flush(stdout)

# continuity check + correction magnitude
function checks()
    maxres=0.0; sumsc=0.0; nres=0
    rels=Float64[]; recu=NaN
    for j in 2:ny-1, i in 2:nx-1
        active(i,j) || continue
        corr=0.0
        for (ni,nj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            (active(ni,nj)||isedge(ni,nj)) || continue   # Neumann faces (land/isolated) carry no correction
            corr += facew(i,j,ni,nj)*(φ[i,j]-φ[ni,nj])   # φ=0 for edge
        end
        Dcorr=Draw[i,j]+corr
        sc=max(abs(T[i,j]),abs(Draw[i,j]),1e-30)
        maxres=max(maxres, abs(Dcorr-T[i,j])/sc); sumsc+=abs(Dcorr-T[i,j])/sc; nres+=1
    end
    # |u'|/|u| per interior x-face (u'=(φ_L-φ_R)/dcen, u=depth-mean raw)
    for j in 2:ny-1, i in 2:nx-1
        (reach[i,j]&&reach[i-1,j]&&wet(i,j)&&wet(i-1,j)) || continue
        ig,jg=i+ng,j+ng; dcen=0.5*(dxo(i-1,j)+dxo(i,j))
        up=(φ[i-1,j]-φ[i,j])/dcen
        # depth-mean raw velocity at this west face
        ubar=0.0; for k in 1:nz; ubar+=u[ig,jg,k]*dsig[k]; end
        push!(rels, abs(up)/max(abs(ubar),0.05))
    end
    (maxres, sumsc/max(nres,1), rels)
end
maxres, meanres, rels=checks()
sort!(rels); q(p)=rels[clamp(round(Int,p*length(rels)),1,length(rels))]
@printf("\nCONTINUITY after projection: max|Dcorr-T|/scale=%.3e  mean=%.3e  (→0 ⇒ closes)\n", maxres, meanres)
@printf("|u'|/|u| over %d interior x-faces: median=%.3f p90=%.3f p99=%.3f max=%.3f\n",
        length(rels), median(rels), q(0.90), q(0.99), rels[end])
println(maxres<1e-8 ? "PASS: continuity closes to ~machine zero." : "CHECK: continuity residual not tiny.")
println(median(rels)<0.3 ? "PASS: correction is gentle (median |u'|/|u|<0.3)." : "WARN: correction large.")
