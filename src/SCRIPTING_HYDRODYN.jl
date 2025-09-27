using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots

run_all_tests();
run_integration_tests()

# TODO implement changes to the hydro module 
#       it needs to accept real NCD files
#       gen grid
# implemented changes to the struct 
netcdf_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"

# The variable map for this specific file (standard ROMS names)
variable_map = Dict(
    :u => "u",
    :v => "v",
    :temp => "temp",
    :salt => "salt",
    :time => "ocean_time"
)
hydro_data = HydrodynamicData(netcdf_filepath, variable_map);

# Simulation time parameters
start_time_seconds = 0.0
end_time_seconds = 24 * 3600.0 # Simulate for 12 hours
dt_seconds = 60.0*30.0 

# A single, continuously active point source
source_config = [
    PointSource(i=100, j=300, k=1, tracer_name=:C_virus, influx_rate=(time)->150000.0),
    PointSource(i=200, j=300, k=1, tracer_name=:C_virus, influx_rate=(time)->150000.0),
]

# --- 3. Initialization ---
println("Opening remote NetCDF file via OPeNDAP...")
ds = NCDataset(netcdf_filepath)

# Get grid dimensions from the dataset
nx = ds.dim["xi_rho"]
ny = ds.dim["eta_rho"]
nz = 1 # We are only simulating the surface layer for this visualization

# Initialize grid and state
grid = initialize_grid(nx, ny, nz, Float64(nx*160), Float64(ny*160), 10.0) # Assume 160m grid spacing
state = initialize_state(grid, (:C_virus,))

println("Initialization complete.")

# --- 4. Run the Simulation ---
println("Starting simulation...")
results, timesteps = run_and_store_simulation(grid, state, source_config, ds, hydro_data, start_time_seconds, end_time_seconds, dt_seconds, 3600.0)

# --- 5. Cleanup ---
close(ds)
println("Simulation finished.")
println("Stored $(length(results)) output steps.")

# --- 6. Visualize Results in the Terminal ---
println("Starting visualization... (Press Ctrl+C to exit)")
max_conc = maximum(maximum(r.tracers[:C_virus]) for r in results if !isempty(r.tracers[:C_virus]))
max_conc = max(max_conc, 1e-9)

for (i, t) in enumerate(timesteps)
    print("\u001b[2J\u001b[H") # Clear screen

    current_hours = round(t / 3600, digits=1)
    current_state = results[i]
    
    # --- NEW: Debugging Probes ---
    # Check the max velocity and concentration at each saved step
    max_u_vel = maximum(abs, current_state.u)
    max_v_vel = maximum(abs, current_state.v)
    max_conc = maximum(current_state.tracers[:C_virus])
    
    println("--- Time: $(current_hours) hours ---")
    println("Max U-Velocity: $(max_u_vel)")
    println("Max V-Velocity: $(max_v_vel)")
    println("Max Concentration: $(max_conc)")
    println("------------------------------------")
    
    # Create the heatmap
    tracer_data = current_state.tracers[:C_virus][:, :, 1]
    plt = heatmap(tracer_data',
                  title="Virus Plume at t = $(current_hours) hours",
                  colormap=:viridis,
                  width = 50,
                  height = 50,
                  zlim=(0, max(max_conc, 1e-9))) # Use dynamic zlim for now
    
    println(plt)
    sleep(0.1) # Pause for a second to read the debug info
end

println("Visualization complete.")

maximum(results[4].tracers[:C_virus])