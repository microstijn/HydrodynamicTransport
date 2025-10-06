using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots
using NCDatasets

f = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"
f = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
hydro_data = create_hydrodynamic_data_from_file(f)

# lets get the grid going
ds = NCDataset(f);
grid = initialize_curvilinear_grid(f);
state = initialize_state(grid, ds, (:Tracer,));

# lets get some sources in Helper
nantes_lon, nantes_lat = -1.549464, 47.197319
source_i, source_j = lonlat_to_ij(grid, nantes_lon, nantes_lat)

sources = [PointSource(i=20, j=30, k=1, tracer_name=:Tracer, influx_rate=(t)->1.0e10)]
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

start_time = 0.0 # Start from the beginning of the dataset
dt = 6.0
end_time = 30 * 60 * 1.0 # Run for 12 hours to keep the test quick

# --- Output Configuration ---
# Directory where the output .jld2 files will be saved
out = raw"D:\PreVir\test_states"

# How often to save the state, in simulation seconds (e.g., 3600.0 for every hour)
out_interval_sec = 30*10.0

# --- Restart Configuration ---
# To restart a simulation, set this to the path of a saved state file.
# For example: const RESTART_FILE = "norway_output/state_t_43200.jld2"
# To start a new simulation, set this to `nothing`.
restart_file = nothing

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    boundary_conditions=bcs,
    advection_scheme=:TVD,
    output_dir=out,
    output_interval=out_interval_sec,
    restart_from=restart_file
)

70heatmap(
    final_state
)