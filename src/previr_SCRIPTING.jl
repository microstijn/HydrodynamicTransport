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
dt = 180
end_time = 48 * 60 * 60.0 # Run for 12 hours to keep the test quick

# --- Output Configuration ---
# Directory where the output .jld2 files will be saved
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
        tau_d = 0.05,       # Critical shear stress for deposition,
        settling_scheme = :BackwardEuler
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

function implicit_adsorption_desorption(concentrations, environment, dt)
    # Unpack concentrations for readability
    C_diss_old = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb_old = max(0.0, concentrations[:Virus_Sorbed])
    
    # Get TSS from the environment state
    TSS = environment.TSS

    # Parameters for the reaction kinetics
    Kd = 0.2              # Partition coefficient (m^3/g)
    transfer_rate = 0.0001  # Rate at which equilibrium is approached (1/s)

    # --- IMPLICIT SOLUTION ---
    # This method is unconditionally stable and suitable for large time steps.
    
    # Total concentration in the cell is a conserved quantity
    C_total = C_diss_old + C_sorb_old
    
    # If there's no mass, no change can occur.
    if C_total <= 0.0
        return Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0)
    end
    
    # Pre-calculate common terms for the implicit formula
    alpha = dt * transfer_rate
    beta = Kd * TSS
    
    # Analytically solve for C_sorb_new using the implicit formula:
    # C_sorb_new = (C_sorb_old + dt*rate*Kd*TSS*C_total) / (1 + dt*rate*(1+Kd*TSS))
    numerator = C_sorb_old + alpha * beta * C_total
    denominator = 1.0 + alpha * (1.0 + beta)
    
    C_sorb_new = numerator / denominator
    
    # The change in sorbed concentration is the difference
    delta_C = C_sorb_new - C_sorb_old
    
    # Return the change for each tracer, maintaining the required interface.
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end


# Wrap the function in the FunctionalInteraction struct
virus_interaction = FunctionalInteraction(
    affected_tracers = [:Virus_Dissolved, :Virus_Sorbed],
    interaction_function = implicit_adsorption_desorption
)

# --- 6. Simulation and Output Parameters ---
start_time = 0.0
end_time = 10*60*60.0 # Run for 48 hours
dt = 60.0

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# IMPORTANT: Update this path to your desired output location.
output_directory = raw"D:\PreVir\test_states_sorb"
output_interval_seconds = 60 * 300.0 # Save output every hour

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

function check_for_nans(state, step_name, time)
    for (name, C) in state.tracers
        if any(isnan, C)
            println("\n" * "="^80)
            println("ERROR: NaN detected in tracer '$name' AFTER step '$step_name' at time t = $time")
            nan_indices = findfirst(isnan, C)
            println("First NaN found at index: $nan_indices")
            println("="^80)
            return true # NaN found
        end
    end
    for (name, B) in state.bed_mass
        if any(isnan, B)
            println("\n" * "="^80)
            println("ERROR: NaN detected in bed_mass for '$name' AFTER step '$step_name' at time t = $time")
            nan_indices = findfirst(isnan, B)
            println("First NaN found at index: $nan_indices")
            println("="^80)
            return true # NaN found
        end
    end
    return false # No NaNs
end

using ProgressMeter
println("\n--- STARTING SIMULATION IN DEBUG MODE ---")
time_range = start_time:dt:end_time


using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HydrodynamicsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule

start_time = 0.0
end_time = 48*60*60.0
dt = 320.0
advection_scheme = :ImplicitADI
D_crit = 0.05
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

@showprogress "Simulating..." for (step, time) in enumerate(time_range)
    if time == start_time; continue; end
    state.time = time

    # --- Run each physics step and check for NaNs immediately after ---

    apply_boundary_conditions!(state, grid, bcs)
    if check_for_nans(state, "apply_boundary_conditions!", time); break; end

    update_hydrodynamics!(state, grid, ds, hydro_data, time)
    if check_for_nans(state, "update_hydrodynamics!", time); break; end

    horizontal_transport!(state, grid, dt, advection_scheme, D_crit, bcs)
    if check_for_nans(state, "horizontal_transport!", time); break; end

    vertical_transport!(state, grid, dt, sediment_params_dict, D_crit)
    if check_for_nans(state, "vertical_transport!", time); break; end

    source_sink_terms!(state, grid, sources, [virus_interaction], time, dt, D_crit)
    if check_for_nans(state, "source_sink_terms!", time); break; end

    # Final positivity constraint
    for C in values(state.tracers)
        C .= max.(0.0, C)
    end
    if check_for_nans(state, "positivity_constraint", time); break; end
end












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

for s in sources_to_plot
    i, j = lonlat_to_ij(grid, s.lon, s.lat)
    if i !== nothing && j !== nothing
        push!(
            sources,
            PointSource(i=i, j=j, k=1, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry=true)
        )
    end
end

# --- 4. Define Sediment Parameters for the Sorbed Tracer ---
println("Defining sediment parameters for :Virus_Sorbed...")
# These parameters control how the sorbed virus settles and resuspends.
sediment_params_dict = Dict(
    :Virus_Sorbed => SedimentParams(
        ws0 = 0.0005,      # Settling velocity in m/s (e.g., 0.5 mm/s)
        tau_cr = 0.1,      # Critical shear stress for erosion
        tau_d = 0.05,      # Critical shear stress for deposition
        settling_scheme = :BackwardEuler # Use the unconditionally stable implicit bed exchange
    )
)

# --- 5. Define the Adsorption/Desorption Functional Interaction ---
println("Defining adsorption-desorption interaction function...")
# This function calculates the mass transfer between dissolved and sorbed forms at each grid cell, each time step.
function implicit_adsorption_desorption(concentrations, environment, dt)
    C_diss_old = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb_old = max(0.0, concentrations[:Virus_Sorbed])
    TSS = environment.TSS; Kd = 0.2; transfer_rate = 0.0001
    C_total = C_diss_old + C_sorb_old
    if C_total <= 0.0; return Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0); end
    alpha = dt * transfer_rate; beta = Kd * TSS
    numerator = C_sorb_old + alpha * beta * C_total
    denominator = 1.0 + alpha * (1.0 + beta)
    C_sorb_new = numerator / denominator
    delta_C = C_sorb_new - C_sorb_old

    # This check is critical to prevent the reaction from creating negative mass
    if delta_C > 0 # Adsorption (dissolved -> sorbed)
        delta_C = min(delta_C, C_diss_old)
    else # Desorption (sorbed -> dissolved)
        delta_C = max(delta_C, -C_sorb_old)
    end
    
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end

virus_interaction = FunctionalInteraction(
    affected_tracers = [:Virus_Dissolved, :Virus_Sorbed],
    interaction_function = implicit_adsorption_desorption
)

# --- 6. Simulation and Output Parameters ---
start_time = 0.0
end_time = 10*60*60.0 # Run for 12 hours
dt = 360.0           # Use a large timestep, enabled by the implicit schemes

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# IMPORTANT: Update this path to your desired output location.
output_directory = raw"D:\PreVir\test_states_sorb"
output_interval_seconds = 30 * 60.0 # Save output every 30 minutes

# --- 7. Run the Simulation ---
println("Starting simulation for $(end_time / 3600.0) hours...")
println("Using a large dt=$dt s, enabled by fully implicit transport.")
println("Output will be saved to: $output_directory")

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    # --- Pass the physics and configuration to the simulation ---
    boundary_conditions = bcs,
    sediment_tracers = sediment_params_dict,
    functional_interactions = [virus_interaction],
    advection_scheme = :ImplicitADI, # Use the unconditionally stable horizontal advection
    # --- Other parameters ---
    D_crit = 0.05,
    output_dir = output_directory,
    output_interval = output_interval_seconds,
    restart_from = nothing
)

# --- 8. Clean Up ---
close(ds)

println("\n--- Simulation Complete ---")
println("Final simulation time: $(final_state.time) seconds.")
println("Total dissolved virus mass: ", sum(final_state.tracers[:Virus_Dissolved] .* grid.volume))
println("Total sorbed virus mass in water: ", sum(final_state.tracers[:Virus_Sorbed] .* grid.volume))
println("Total sorbed virus mass in bed: ", sum(final_state.bed_mass[:Virus_Sorbed]))
