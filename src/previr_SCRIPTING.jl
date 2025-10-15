using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots
using NCDatasets


f = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"
hydro_data = create_hydrodynamic_data_from_file(f)

# lets get the grid going
ds = NCDataset(f);
grid = initialize_curvilinear_grid(f);
state = initialize_state(grid, ds, (:Tracer,));


st = estimate_stable_timestep(hydro_data,
                                 advection_scheme=:ImplicitADI,
                                 dx_var ="dx", 
                                 dy_var ="dy",  
                                 safety_factor=0.8,
                                 CFL_acc = 100.0,
                                 time_samples = 3
)

# lets get some sources in Helper

sources_to_plot = [
    (name = "Nantes",         lon = -1.549464,  lat = 47.197319),
    (name = "Saint-Nazaire",  lon = -2.28,      lat = 47.27),
    (name = "Cordemais",      lon = -1.97,      lat = 47.28)
]

sources = PointSource[]

for s in sources_to_plot
    i, j = lonlat_to_ij(grid, s.lon, s.lat)
    push!(
        sources,
        PointSource(i = i, j = j , k=1, tracer_name=:Tracer, influx_rate=(t)->1.0e10, relocate_if_dry = true)
    )
end

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]



start_time = 0.0 # Start from the beginning of the dataset
dt = 6.0
end_time = 48 * 60 * 60.0 # Run for 12 hours to keep the test quick

# --- Output Configuration ---
# Directory where the output .jld2 files will be saved
out = raw"D:\PreVir\test_states"
out = raw"D:\PreVir\test_states"

# How often to save the state, in simulation seconds (e.g., 3600.0 for every hour)
out_interval_sec = 30*60.0

# --- Restart Configuration ---
# To restart a simulation, set this to the path of a saved state file.
# For example:  RESTART_FILE = "norway_output/state_t_43200.jld2"
# To start a new simulation, set this to `nothing`.
restart_file = nothing

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    boundary_conditions = bcs,
    advection_scheme = :TVD,
    D_crit = 0.05,
    output_dir = out,
    output_interval = out_interval_sec,
    restart_from = restart_file
)

tracer_phys = view(
    final_state.tracers[:Tracer],
    grid.ng+1:grid.nx+grid.ng,
    grid.ng+1:grid.ny+grid.ng,
    1
)

println(
    heatmap(
        tracer_phys',
        title="Final Tracer Concentration (Console)",
        colormap=:viridis,
        labels=false,
        colorbar= true,
        width = 50)
    )
contourplot(tracer_phys', colorbar= true, width = 50, height = 50)







