# run_loire_simulation.jl

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
tracer_names = (:Virus_Dissolved, :Virus_Sorbed)
# Specify that :Virus_Sorbed has a corresponding bed_mass array
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list)

# Set a uniform background TSS concentration (e.g., 5.0 g/m^3)
# A real simulation might read this from the NetCDF file if available.
state.tss .= 5.0

# --- 3. Source Configuration ---
# The source only introduces the DISSOLVED form of the virus.
println("Configuring point source for dissolved virus...")
sources = PointSource[]
source_locations = [
    (name = "Nantes",        lon = -1.549464,  lat = 47.197319),
    (name = "Saint-Nazaire", lon = -2.28,      lat = 47.27),
    (name = "Cordemais",     lon = -1.97,      lat = 47.28)
]

for loc in source_locations
    i, j = lonlat_to_ij(grid, loc.lon, loc.lat)
    if i !== nothing && j !== nothing
        println("  -> Source '$(loc.name)' placed at grid indices (i=$i, j=$j)")
        push!(
            sources,
            PointSource(i=i, j=j, k=grid.nz, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry=true)
        )
    else
        println("  -> Warning: Could not find grid indices for source '$(loc.name)'.")
    end
end

# --- 4. Define Sediment Parameters for the Sorbed Tracer ---
println("Defining sediment parameters for :Virus_Sorbed...")
# NOTE: These parameters have been adapted to the `SedimentParams` struct in ModelStructs.jl
sediment_params = Dict(
    :Virus_Sorbed => SedimentParams(
        ws = 0.0005,           # Settling velocity (ws) in m/s (e.g., 0.5 mm/s)
        erosion_rate = 1e-7,   # Constant erosion rate in kg/m^2/s (a value for the simple bed model)
        tau_ce = 0.1           # Critical shear stress for erosion (Pa), for future, more advanced bed models
    )
)

# --- 5. Define the Adsorption/Desorption Functional Interaction ---
println("Defining adsorption-desorption interaction function...")
# This function calculates the mass transfer between dissolved and sorbed forms.
function implicit_adsorption_desorption(concentrations, environment, dt)
    C_diss_old = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb_old = max(0.0, concentrations[:Virus_Sorbed])
    TSS = environment.TSS; Kd = 0.2; transfer_rate = 0.0001
    
    C_total = C_diss_old + C_sorb_old
    if C_total <= 1e-12; return Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0); end
    
    alpha = dt * transfer_rate; beta = Kd * TSS
    
    # Implicit solution for the new sorbed concentration
    numerator = C_sorb_old + alpha * beta * C_total
    denominator = 1.0 + alpha * (1.0 + beta)
    C_sorb_new = numerator / denominator
    
    delta_C = C_sorb_new - C_sorb_old

    # This check is critical to prevent the reaction from creating/destroying mass
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
functional_interactions = [virus_interaction]

# --- 6. Simulation and Output Parameters ---
start_time = 0.0
end_time = 1260.0 # Run for 12 hours
dt = 6.0             # Use a large timestep, enabled by the implicit schemes

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# IMPORTANT: Update this path to your desired output location.
output_directory = raw"D:\PreVir\loire_virus_sim_output"
output_interval_seconds = 30 * 60.0 # Save output every 30 minutes

# --- 7. Run the Simulation ---
println("\n--- Starting simulation ---")
println("Total duration: $(end_time / 3600.0) hours")
println("Time step (dt): $dt seconds")
println("Output will be saved to: $output_directory")

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    boundary_conditions = bcs,
    sediment_params = sediment_params,
    functional_interactions = functional_interactions,
    advection_scheme = :TVD,
    D_crit           = 0.05,
    output_dir       = output_directory,
    output_interval = output_interval_seconds,
    restart_from = nothing
)

# --- 8. Clean Up and Summarize ---
close(ds)

println("\n--- Simulation Complete ---")
println("Final simulation time: $(round(final_state.time / 3600.0, digits=2)) hours.")

total_dissolved_mass = sum(final_state.tracers[:Virus_Dissolved])
total_sorbed_water_mass = sum(final_state.tracers[:Virus_Sorbed])
total_bed_mass = sum(final_state.bed_mass[:Virus_Sorbed])
using UnicodePlots

heatmap(
    final_state.tracers[:Virus_Sorbed][:, :, 1],
    width = 50
)

println("Total dissolved virus mass: ", total_dissolved_mass)
println("Total sorbed virus mass in water: ", total_sorbed_water_mass)
println("Total sorbed virus mass in bed: ", total_bed_mass)
println("Total virus mass in system: ", total_dissolved_mass + total_sorbed_water_mass + total_bed_mass)





