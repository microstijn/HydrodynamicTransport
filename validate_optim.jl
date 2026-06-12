# Regression harness for the #1 (hydro slab cache) + #2 (no per-step deepcopy) optimizations.
#
# Uses the REAL CurviLoire 2015 curvilinear grid + real NetCDF data path (the actual
# production path — NOT a Cartesian/placeholder grid, and the only path on which #1 runs),
# but only a short 1-hour run with 2 tracers and no state output. Prints a high-precision
# checksum of the final tracer fields. Run before and after the change; checksums must be
# identical to machine precision.
using Pkg
# Use the softMode environment (it precompiles cleanly on this nightly and depends on
# HydrodynamicTransport via a path dep -> loads this working tree / current branch).
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.TimeSteppingModule
using NCDatasets
using Printf

const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"
isfile(NC) || error("CurviLoire 2015 file not found: $NC")

println("Loading curvilinear grid + hydro data from real CurviLoire 2015 file...")
grid = initialize_curvilinear_grid(NC)
hydro_data = create_hydrodynamic_data_from_file(NC)
ds = NCDataset(NC)
state = initialize_state(grid, ds, (:A, :B))

# Two repaired/validated wet source cells (from the V10 set): coastal NW shore + far upstream.
sources = [
    PointSource(i=156, j=156, k=1, tracer_name=:A, influx_rate=t->1.0e6, relocate_if_dry=true),
    PointSource(i=411, j=72,  k=1, tracer_name=:B, influx_rate=t->1.0e6, relocate_if_dry=true),
]

# Short run: 1 hour from the S1/W1 release time, adaptive dt, same numerics as the campaign.
start_t = 10_886_400.0          # ~2015-05-07 00:00 on the annual time axis
end_t   = start_t + 3600.0      # +1 hour (crosses ≥1 hydro interval; many sub-steps)
final = run_simulation(grid, state, sources, start_t, end_t, 300.0;
    ds=ds, hydro_data=hydro_data, use_adaptive_dt=true,
    cfl_max=0.9, dt_max=1500.0, dt_min=0.01, dt_growth_factor=1.1,
    advection_scheme=:TVD, D_crit=0.05, write_full_state=false)
close(ds)

a = sum(final.tracers[:A]); b = sum(final.tracers[:B])
@printf("CHECKSUM sumA=%.17e sumB=%.17e maxA=%.17e maxB=%.17e t=%.6f\n",
        a, b, maximum(final.tracers[:A]), maximum(final.tracers[:B]), final.time)
