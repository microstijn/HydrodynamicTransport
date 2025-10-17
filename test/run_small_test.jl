# run_small_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using NCDatasets

println("--- HydrodynamicTransport.jl: Small Dataset Test ---")

# --- 1. Data Configuration ---
# This special OPeNDAP URL requests a small subset of the large Norway dataset.
# It asks for:
#   - Time steps 0 through 4 (5 total)
#   - Vertical layers 30 through 34 (top 5 layers)
#   - A 50x50 horizontal patch (indices 80-129 for 'i' and 280-329 for 'j')
subset_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc?u[0:1:4][30:1:34][280:1:329][80:1:129],v[0:1:4][30:1:34][280:1:329][80:1:129],ocean_time[0:1:4],lon_rho[280:1:329][80:1:129],lat_rho[280:1:329][80:1:129],lon_u[280:1:329][80:1:129],lat_u[280:1:329][80:1:129],lon_v[280:1:329][80:1:129],lat_v[280:1:329][80:1:129],pm[280:1:329][80:1:129],pn[280:1:329][80:1:129],angle[280:1:329][80:1:129],h[280:1:329][80:1:129],mask_rho[280:1:329][80:1:129],mask_u[280:1:329][80:1:129],mask_v[280:1:329][80:1:129],zeta[0:1:4][280:1:329][80:1:129],s_rho,s_w,hc,Cs_r,Cs_w"

println("Using small remote subset: $subset_filepath")
hydro_data = create_hydrodynamic_data_from_file(subset_filepath)

# --- 2. Grid and State Initialization ---
println("Connecting to NetCDF subset...")
ds = NCDataset(subset_filepath)

println("Initializing Curvilinear Grid (50x50x5)...")
grid = initialize_curvilinear_grid(subset_filepath)

println("Initializing State...")
tracer_names = (:TestTracer,)
state = initialize_state(grid, ds, tracer_names)

# --- 3. Source and Simulation Parameters ---
# Place a single source in the middle of our new, small 50x50 grid.
# The grid indices are now relative to the subset.
sources = [
	PointSource(i=25, j=25, k=grid.nz, tracer_name=:TestTracer, influx_rate=(t)->1.0e6)
]

# Get the time range directly from the subset file
time_dim = ds["ocean_time"]
start_time = time_dim[1]
end_time = time_dim[end]
dt = (time_dim[2] - time_dim[1]) / 2.0 # Use a reasonable timestep

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# --- 4. Run the Simulation ---
println("\n--- Starting short simulation on subset ---")
println("Start time: $start_time s, End time: $end_time s, dt: $dt s")

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    boundary_conditions = bcs,
    advection_scheme = :TVD
)

# --- 5. Clean Up and Verify ---
close(ds)

final_mass = sum(final_state.tracers[:TestTracer] .* grid.volume)

println("\n--- Simulation Complete ---")
if final_mass > 0
    println("✅ Test successful! Final tracer mass is greater than zero.")
else
    println("❌ Test failed! Final tracer mass is zero.")
end
println("Final mass: ", final_mass)