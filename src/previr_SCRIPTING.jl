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
state = initialize_state(grid, ds, (:tracer_d,););


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









# run_loire_sim.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using NCDatasets


println("--- HydrodynamicTransport.jl: Loire Estuary Sorption & Sedimentation Simulation ---")

# --- 1. Data Configuration ---
# IMPORTANT: Update this path to the location of your NetCDF file.
loire_filepath = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"

println("Autodetecting variables from: $loire_filepath")
hydro_data = create_hydrodynamic_data_from_file(loire_filepath)

# --- 2. Grid and State Initialization ---
println("Connecting to NetCDF file...")
ds = NCDataset(loire_filepath)

println("Initializing Curvilinear Grid...")
grid = initialize_curvilinear_grid(loire_filepath)

println("Initializing State with Dissolved and Sorbed tracers...")
# Define all tracers for the simulation
tracer_names = (:Virus_Dissolved, :Virus_Sorbed,)
# Specify that :Virus_Sorbed has a bed component
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list)
state.tss
# Set a uniform background TSS concentration (e.g., 5.0 g/m^3)
# A real simulation might read this from the NetCDF file if available.
state.tss .= 5.0

# --- 3. Source Configuration ---
# The source only introduces the DISSOLVED form of the virus.
println("Configuring point source for dissolved virus...")
sources = PointSource[]
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
        PointSource(i = i, j = j , k=1, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry = true)
    )
end

# --- 4. Define Sediment Parameters for the Sorbed Tracer ---
println("Defining sediment parameters for :Virus_Sorbed...")
# These parameters control how the sorbed virus settles and resuspends.
# We'll give it a slow settling velocity (ws0).
sediment_params_dict = Dict(
    :Virus_Sorbed => SedimentParams(
        ws0 = 0.0005,      # Settling velocity in m/s (e.g., 0.5 mm/s)
        tau_cr = 0.1,      # Critical shear stress for erosion
        tau_d = 0.05       # Critical shear stress for deposition
    )
)

# --- 5. Define the Adsorption/Desorption Functional Interaction ---
println("Defining adsorption-desorption interaction function...")
# This function calculates the mass transfer between dissolved and sorbed forms at each grid cell, each time step.
function adsorption_desorption(concentrations, environment, dt)
    # Unpack concentrations for readability
    C_diss = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb = max(0.0, concentrations[:Virus_Sorbed])
    
    # Get TSS from the environment state
    TSS = environment.TSS

    # Parameters for the reaction kinetics
    Kd = 0.2              # Partition coefficient (m^3/g)
    transfer_rate = 0.0001  # Rate at which equilibrium is approached (1/s)

    # Calculate the equilibrium concentration for the sorbed phase
    C_sorb_eq = Kd * TSS * C_diss

    # Calculate the change based on the difference from equilibrium
    # This is a simple first-order approach to equilibrium
    delta_C = (C_sorb_eq - C_sorb) * transfer_rate * dt

    # Ensure we don't transfer more mass than is available
    if delta_C > 0 # Adsorption (dissolved -> sorbed)
        delta_C = min(delta_C, C_diss)
    else # Desorption (sorbed -> dissolved)
        delta_C = max(delta_C, -C_sorb)
    end
    
    # Return the CHANGE for each tracer. Note the signs.
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end

# Wrap the function in the FunctionalInteraction struct
virus_interaction = FunctionalInteraction(
    affected_tracers = [:Virus_Dissolved, :Virus_Sorbed],
    interaction_function = adsorption_desorption
)

# --- 6. Simulation and Output Parameters ---
start_time = 0.0
end_time = 12*60*60.0 # Run for 48 hours
dt = 60.0

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# IMPORTANT: Update this path to your desired output location.
output_directory = raw"D:\PreVir\test_states_sorb"
output_interval_seconds = 30 * 60.0 # Save output every hour

# --- 7. Run the Simulation ---
println("Starting simulation for $(end_time / 3600.0) hours...")
println("Output will be saved to: $output_directory")

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    boundary_conditions = bcs,
    # --- Pass the new physics modules to the simulation ---
    sediment_tracers = sediment_params_dict,
    functional_interactions = [virus_interaction],
    # --- Other parameters ---
    advection_scheme = :ImplicitADI,
    D_crit = 0.05,
    output_dir = output_directory,
    output_interval = output_interval_seconds,
    restart_from = nothing
)

final_state.tracers
sum(final_state.tracers[:Virus_Dissolved])
sum(final_state.tracers[:Virus_Sorbed])
sum(final_state.bed_mass[:Virus_Sorbed])


# --- 8. Clean Up ---
close(ds)

println("\n--- Simulation Complete ---")
println("Final simulation time: $(final_state.time) seconds.")