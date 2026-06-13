# Baseline benchmark + profile of horizontal_transport! (:FFSL) on the REAL CurviLoire grid.
# Establishes a stable per-step min (over batches) for 8 and 20 tracers, then profiles to
# surface the actual hot lines in the FFSL kernels before any micro-optimization.
# Run:  JULIA_NUM_THREADS=8 julia +nightly claude/bench_ffsl.jl
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.HydrodynamicsModule
using NCDatasets
using Printf
using Profile

const HT  = HydrodynamicTransport
const HTM = HydrodynamicTransport.HorizontalTransportModule

const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"
const START_T = 10_886_400.0
const DT = 60.0
const D_CRIT = 0.05
const SCHEME = :FFSL

println("threads = ", Threads.nthreads())
println("Loading real CurviLoire grid + hydro ...")
grid = initialize_curvilinear_grid(NC)
hydro_data = create_hydrodynamic_data_from_file(NC)
ds = NCDataset(NC)

function make_state(ntr::Int)
    names = ntuple(i -> Symbol("T", i), ntr)
    st = initialize_state(grid, ds, names)
    update_hydrodynamics!(st, grid, ds, hydro_data, START_T)
    # seed a couple of blobs so the FCT limiter sees real gradients
    nx, ny, nz = grid.nx, grid.ny, grid.nz; ng = grid.ng
    for nm in names
        C = st.tracers[nm]
        for k in 1:nz, j in 1:ny, i in 1:nx
            ig, jg = i+ng, j+ng
            r2 = (i - nx*0.4)^2 + (j - ny*0.5)^2
            C[ig, jg, k] = Float32(exp(-r2 / (2*30.0^2)))
        end
    end
    return st
end

bcs = HT.BoundaryCondition[]

function bench(ntr::Int; nbatch=6, nper=10)
    st = make_state(ntr)
    HTM.horizontal_transport!(st, grid, DT, SCHEME, D_CRIT, bcs)  # warmup/compile
    best = Inf
    for _ in 1:nbatch
        t = @elapsed for _ in 1:nper
            HTM.horizontal_transport!(st, grid, DT, SCHEME, D_CRIT, bcs)
        end
        best = min(best, t/nper)
    end
    @printf("ntr=%2d : %.2f ms/step (min over %d batches of %d)\n", ntr, best*1e3, nbatch, nper)
    return best, st
end

println("\n--- baseline timing (tracer-count sweep) ---")
b1,  _  = bench(1)
b8,  _  = bench(8)
b16, _  = bench(16)
b20, st = bench(20)
b24, _  = bench(24)
@printf("\nper-step / (ntr/8 rounds):  8->%.1f  16->%.1f  20->%.1f  24->%.1f ms/round\n",
        b8*1e3/1, b16*1e3/2, b20*1e3/3, b24*1e3/3)

println("\n--- profile (20 tracers, 40 steps) ---")
Profile.clear()
Profile.@profile for _ in 1:40
    HTM.horizontal_transport!(st, grid, DT, SCHEME, D_CRIT, bcs)
end
Profile.print(format=:flat, sortedby=:count, mincount=30)
close(ds)
