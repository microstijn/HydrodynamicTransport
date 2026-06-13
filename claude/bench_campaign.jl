# Campaign runtime estimate: time a short REAL run_simulation (20 tracers, FFSL, campaign numerics)
# on the real CurviLoire grid, then extrapolate to the full 21 windows x 4 weeks x ~20 tracers.
# Measures the authoritative wall/sim ratio (transport + vertical + sources + hydro slab I/O +
# adaptive dt + the output/receptor boundary clamp is excluded here -> noted as a small extra).
#   JULIA_NUM_THREADS=8 julia +nightly claude/bench_campaign.jl
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs, HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule, HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.TimeSteppingModule
using NCDatasets, Printf
const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"

const NTR   = 20
const START = 10_886_400.0
# campaign numerics (K7): TVD->FFSL, cfl=0.9, dt_max=1500, dt_min=0.01, D_crit=0.05
const NUM = (use_adaptive_dt=true, cfl_max=0.9, dt_max=1500.0, dt_min=0.01,
             dt_growth_factor=1.1, advection_scheme=:FFSL, D_crit=0.05, write_full_state=false)

println("threads = ", Threads.nthreads(), " ; tracers = ", NTR)
grid = initialize_curvilinear_grid(NC)
hydro = create_hydrodynamic_data_from_file(NC)
ds = NCDataset(NC)
names = ntuple(i -> Symbol("T", i), NTR)
state = initialize_state(grid, ds, names)

# a few wet point sources (validated V10 cells); transport processes ALL 20 tracers regardless
sources = PointSource[
    PointSource(i=156, j=156, k=1, tracer_name=:T1, influx_rate=t->1.0e6, relocate_if_dry=true),
    PointSource(i=411, j=72,  k=1, tracer_name=:T2, influx_rate=t->1.0e6, relocate_if_dry=true),
    PointSource(i=156, j=156, k=1, tracer_name=:T3, influx_rate=t->1.0e6, relocate_if_dry=true),
    PointSource(i=411, j=72,  k=1, tracer_name=:T4, influx_rate=t->1.0e6, relocate_if_dry=true),
]

run1(t0, t1) = run_simulation(grid, deepcopy(state), sources, t0, t1, 60.0;
                              ds=ds, hydro_data=hydro, NUM...)

println("warmup (compile + cache slabs): 20 min sim ...")
run1(START, START + 1200.0)

const SIM = 3 * 3600.0   # timed window: 3 h sim
println("timed run: ", SIM/3600, " h sim ...")
wall = @elapsed run1(START, START + SIM)

ratio = wall / SIM                      # wall seconds per simulated second
total_sim = 21 * 4 * 7 * 86400.0        # 21 windows x 4 weeks x 7 days
total_wall = ratio * total_sim
per_run_wall = ratio * (4 * 7 * 86400.0)

@printf("\nTIMED: %.1f s wall for %.0f s sim  ->  wall/sim = %.4f  (%.2f wall-h per sim-day)\n",
        wall, SIM, ratio, ratio*86400/3600)
# NOTE: FFSL adaptive dt here runs ~26-56 s (gradient-CFL bound on this estuary grid; see notes),
# NOT 60 s. The wall/sim ratio is dt-independent, so the extrapolation below is unaffected.
println("="^70)
@printf("CAMPAIGN = 21 windows x 4 weeks x %d tracers  (total sim = %.0f days)\n", NTR, total_sim/86400)
@printf("  per 4-week window : %.1f wall-hours\n", per_run_wall/3600)
@printf("  full 21 windows   : %.1f wall-hours  = %.1f days\n", total_wall/3600, total_wall/86400)
println("(excludes 6-hourly full-state output + hourly receptor I/O -> add a few %.)")
close(ds)
