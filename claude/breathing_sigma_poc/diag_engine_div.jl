# Fair test: use the ENGINE'S OWN grid.face_area_x/y (built in GridModule with face-averaged dy) and the
# exact hdiv from diagnose_vertical_velocity! to compute Dtot, and correlate with -Area·∂η/∂t. This tells
# us whether the ENGINE's horizontal divergence is accurate enough to represent the tidal signal, or is
# reconstruction-dominated (which would mean the real_probe C≡1 error is largely a metric artifact, and the
# breathing fix would not help until the divergence reconstruction is fixed).
import Pkg
const HT=raw"c:\Users\peete074\OneDrive - Wageningen University & Research\programming\HydrodynamicTransport"
Pkg.activate(HT)
using HydrodynamicTransport, HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets, Printf, Statistics
const path=raw"C:\Users\peete074\AppData\Local\Temp\claude\c--Users-peete074-OneDrive---Wageningen-University---Research-programming-softMode\08611189-a493-4304-af89-8d803dd2612b\scratchpad\run_curviloire_2010.nc"

grid=initialize_curvilinear_grid(path); hydro=create_hydrodynamic_data_from_file(path)
ds=NCDataset(path); state=initialize_state(grid, ds,(:C,))
ng,nx,ny,nz=grid.ng,grid.nx,grid.ny,grid.nz
tv=Float64.(ds["time"][:])
fax=grid.face_area_x; fay=grid.face_area_y   # ENGINE's own static face areas

X=Float64[]; Y=Float64[]
for ip in (200,300,400,500,600)
    XEn=coalesce.(ds["XE"][:,:,ip],0.0); XEnp1=coalesce.(ds["XE"][:,:,ip+1],0.0); dtf=tv[ip+1]-tv[ip]
    update_hydrodynamics!(state,grid,ds,hydro,0.5*(tv[ip]+tv[ip+1]); diagnose_w=false)
    u=state.u; v=state.v
    for j in 2:ny-1, i in 2:nx-1
        (grid.mask_rho[i+ng,j+ng]&&grid.h[i+ng,j+ng]>1.0)||continue
        ig,jg=i+ng,j+ng; area=1/(grid.pm[ig,jg]*grid.pn[ig,jg])
        Dtot=0.0
        for k in 1:nz   # exact hdiv from diagnose_vertical_velocity!
            Dtot += u[ig+1,jg,k]*fax[ig+1,jg,k]-u[ig,jg,k]*fax[ig,jg,k]+v[ig,jg+1,k]*fay[ig,jg+1,k]-v[ig,jg,k]*fay[ig,jg,k]
        end
        push!(X,Dtot); push!(Y,-area*(XEnp1[i,j]-XEn[i,j])/dtf)
    end
end
close(ds)
pear(a,b)=(am=mean(a);bm=mean(b); sum((a.-am).*(b.-bm))/sqrt(sum((a.-am).^2)*sum((b.-bm).^2)))
slp(a,b)=(am=mean(a); sum((a.-am).*(b.-mean(b)))/sum((a.-am).^2))
@printf("N=%d\n", length(X))
@printf("ENGINE Dtot vs -Area∂η/∂t:  r=%.3f  slope=%.3f\n", pear(X,Y), slp(X,Y))
@printf("std(engine Dtot)=%.2f  std(target)=%.2f m³/s  ratio=%.1f×\n", std(X), std(Y), std(X)/std(Y))
thr=quantile(abs.(Y),0.75); idx=findall(t->abs(t)>thr,Y)
@printf("tidally-active 25%%: r=%.3f  slope=%.3f\n", pear(X[idx],Y[idx]), slp(X[idx],Y[idx]))
println("\nratio≈1 & r≈1 => engine divergence is accurate (breathing signal resolvable);")
println("ratio≫1 & r≈0 => engine divergence is reconstruction-dominated (the real issue upstream of breathing).")
