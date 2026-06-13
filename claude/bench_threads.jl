# Thread-scaling of horizontal_transport! (:FFSL): fixed work (8 tracers), vary thread count
# via the process JULIA_NUM_THREADS. Tells us compute-bound (scales ~linearly) vs
# bandwidth/clock-bound (saturates). Run once per thread count:
#   JULIA_NUM_THREADS=N julia +nightly claude/bench_threads.jl
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs, HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule, HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.HydrodynamicsModule
using NCDatasets, Printf
const HT = HydrodynamicTransport
const HTM = HydrodynamicTransport.HorizontalTransportModule
const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"
grid = initialize_curvilinear_grid(NC)
hydro_data = create_hydrodynamic_data_from_file(NC)
ds = NCDataset(NC)
names = ntuple(i -> Symbol("T", i), 8)
st = initialize_state(grid, ds, names)
update_hydrodynamics!(st, grid, ds, hydro_data, 10_886_400.0)
nx, ny, nz = grid.nx, grid.ny, grid.nz; ng = grid.ng
for nm in names, k in 1:nz, j in 1:ny, i in 1:nx
    r2 = (i - nx*0.4)^2 + (j - ny*0.5)^2
    st.tracers[nm][i+ng, j+ng, k] = Float32(exp(-r2/(2*30.0^2)))
end
bcs = HT.BoundaryCondition[]
HTM.horizontal_transport!(st, grid, 60.0, :FFSL, 0.05, bcs)
best = Inf
for _ in 1:8
    t = @elapsed for _ in 1:10
        HTM.horizontal_transport!(st, grid, 60.0, :FFSL, 0.05, bcs)
    end
    global best = min(best, t/10)
end
@printf("threads=%d  8tracers=%.2f ms/step\n", Threads.nthreads(), best*1e3)
close(ds)
