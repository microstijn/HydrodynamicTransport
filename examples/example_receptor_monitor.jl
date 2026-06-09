# examples/example_receptor_monitor.jl

using HydrodynamicTransport

# ---------------------------------------------------------
# Example: Using Generic Receptor Monitoring in a Simulation
# ---------------------------------------------------------
#
# This script demonstrates how to set up receptor monitoring
# to write high-frequency time series data at specific points
# while skipping full-grid model state output.
# ---------------------------------------------------------

# Placeholders for paths and settings
const OUTPUT_DIR = "path/to/output"
const HYDRO_FILE = "path/to/hydro_input.nc"

# Example simulation settings
simulation_start_time_seconds = 0.0
simulation_end_time_seconds   = 24.0 * 3600.0  # 1 day
dt_initial_seconds            = 60.0

# ---------------------------------------------------------
# 1. Setup Grid, State, Sources (Mocks for the example)
# ---------------------------------------------------------
# In a real run, you would load or initialize your real grid.
println("Initializing mock grid and state...")
grid = initialize_cartesian_grid(100, 100, 10, 1000.0, 10.0, 5.0)

tracers = (:TRACER_A, :TRACER_B)
state = initialize_state(grid, tracers)

sources = PointSource[]

# ---------------------------------------------------------
# 2. Setup Receptor Monitors
# ---------------------------------------------------------
# We want to normalize the raw extracted value for each tracer
# by the total mass or some standard normalization constant.
normalization_by_tracer = Dict(
    :TRACER_A => 1.0e12,
    :TRACER_B => 1.0e12,
)

# Example: Receptor A at lon/lat
# Using `create_receptor_monitor_from_lonlat` handles finding
# the correct grid indices `i` and `j` internally.
receptor_a_lon = -1.5
receptor_a_lat = 47.0

# NOTE: If using a simple Cartesian mock grid without real lon/lat coords,
# this function call will fail unless grid.lon_rho and grid.lat_rho exist.
# For the sake of the generic example structure, we show the API as it would
# look with a real geographic grid (e.g., CurvilinearGrid).
#
# monitor_a = HydrodynamicTransport.create_receptor_monitor_from_lonlat(
#     grid;
#     receptor_id = "RECEPTOR_A",
#     lon = receptor_a_lon,
#     lat = receptor_a_lat,
#     tracer_names = [:TRACER_A, :TRACER_B],
#     normalization_by_tracer = normalization_by_tracer,
#     output_file = joinpath(OUTPUT_DIR, "RECEPTOR_A_monitor_1h.csv"),
#     start_time_seconds = simulation_start_time_seconds,
# )

# For this purely abstract Cartesian grid example, we construct the ReceptorMonitor directly:
monitor_a = ReceptorMonitor(
    receptor_id = "RECEPTOR_A",
    i = 50,
    j = 50,
    i_array = 50 + grid.ng,  # account for halo
    j_array = 50 + grid.ng,
    tracer_names = [:TRACER_A, :TRACER_B],
    normalization_by_tracer = normalization_by_tracer,
    output_file = joinpath(OUTPUT_DIR, "RECEPTOR_A_monitor_1h.csv"),
    start_time_seconds = simulation_start_time_seconds
)

monitor_b = ReceptorMonitor(
    receptor_id = "RECEPTOR_B",
    i = 75,
    j = 25,
    i_array = 75 + grid.ng,
    j_array = 25 + grid.ng,
    tracer_names = [:TRACER_A, :TRACER_B],
    normalization_by_tracer = normalization_by_tracer,
    output_file = joinpath(OUTPUT_DIR, "RECEPTOR_B_monitor_1h.csv"),
    start_time_seconds = simulation_start_time_seconds
)

# ---------------------------------------------------------
# 3. Run Simulation with Monitoring
# ---------------------------------------------------------
println("Starting simulation...")

# Here we disable full_state output (write_full_state = false),
# and enable receptor output every 1 hour.
HydrodynamicTransport.run_simulation(
    grid,
    state,
    sources,
    simulation_start_time_seconds,
    simulation_end_time_seconds,
    dt_initial_seconds;

    # Standard output dir handling
    output_dir = OUTPUT_DIR,

    # ---------------------------------------------------
    # NEW RECEPTOR MONITOR ARGS
    # ---------------------------------------------------
    write_full_state = false,
    full_state_output_interval = 24.0 * 3600.0,

    receptor_monitors = [monitor_a, monitor_b],
    receptor_monitor_interval = 1.0 * 3600.0,
    # ---------------------------------------------------

    use_adaptive_dt = true,
    # (other numerical parameters remain unchanged)
)

println("Simulation complete. Receptor CSV files should be written in $OUTPUT_DIR")
