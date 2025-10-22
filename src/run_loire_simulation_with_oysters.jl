# run_loire_simulation.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using NCDatasets
using UnicodePlots

println("--- HydrodynamicTransport.jl: Loire Estuary Sorption, Sedimentation, and Oyster Simulation ---")

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
tracer_names = (:Virus_Dissolved, :Virus_Sorbed)
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list)

# Set a uniform background TSS concentration (e.g., 10.0 mg/L, which is g/m^3)
# A real simulation might read this from the NetCDF file if available.
state.tss .= 10.0
# Also set uniform Temp and Salinity for the oyster model if they are not in the hydro_data
if !haskey(hydro_data.var_map, :temp); state.temperature .= 15.0; end
if !haskey(hydro_data.var_map, :salt); state.salinity .= 25.0; end


# --- 3. Source Configuration ---
println("Configuring point sources for dissolved virus...")
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
        push!(sources, PointSource(i=i, j=j, k=grid.nz, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry=true))
    else
        println("  -> Warning: Could not find grid indices for source '$(loc.name)'.")
    end
end

# --- 4. Define Sediment Parameters for the Sorbed Tracer ---
println("Defining sediment parameters for :Virus_Sorbed...")
sediment_params = Dict(
    :Virus_Sorbed => SedimentParams(ws = 0.0005, erosion_rate = 1e-7, tau_ce = 0.1)
)

# --- 5. Define the Adsorption/Desorption Functional Interaction ---
println("Defining adsorption-desorption interaction function...")
function implicit_adsorption_desorption(concentrations, environment, dt)
    C_diss_old = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb_old = max(0.0, concentrations[:Virus_Sorbed])
    TSS = environment.TSS; Kd = 0.2; transfer_rate = 0.0001
    C_total = C_diss_old + C_sorb_old
    if C_total <= 1e-12; return Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0); end
    alpha = dt * transfer_rate; beta = Kd * TSS
    numerator = C_sorb_old + alpha * beta * C_total
    denominator = 1.0 + alpha * (1.0 + beta)
    C_sorb_new = numerator / denominator
    delta_C = C_sorb_new - C_sorb_old
    if delta_C > 0; delta_C = min(delta_C, C_diss_old); else; delta_C = max(delta_C, -C_sorb_old); end
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end

virus_interaction = FunctionalInteraction(
    affected_tracers = [:Virus_Dissolved, :Virus_Sorbed],
    interaction_function = implicit_adsorption_desorption
)


# decay

function create_decay_interaction(params::DecayParams)
    function decay_function(concentrations, environment, dt)
        C_old = max(0.0, concentrations[params.tracer_name])
        if C_old <= 1e-12; return Dict(params.tracer_name => 0.0); end
        T = environment.T
        k_temp = if params.temp_theta > 1.0 && !isnan(T)
            params.base_rate * params.temp_theta^(T - params.temp_ref)
        else
            params.base_rate
        end
        UVB = environment.UVB
        k_light = if params.light_coeff > 0.0 && !isnan(UVB)
            params.light_coeff * UVB
        else
            0.0
        end
        k_total = k_temp + k_light
        delta_C = -k_total * C_old * dt
        delta_C = max(delta_C, -C_old)
        return Dict(params.tracer_name => delta_C)
    end
    return FunctionalInteraction(
        affected_tracers = [params.tracer_name],
        interaction_function = decay_function
    )
end

decay_params = DecayParams(
    tracer_name = :Virus_Dissolved,
    base_rate = 1.0 / (3 * 24 * 3600.0), # 3-day half-life
    temp_theta = 1.07 # Decay is faster in warmer water
)

decay_interaction = create_decay_interaction(decay_params)

functional_interactions = [virus_interaction, decay_interaction]

# --- 6. Oyster Configuration ---
println("Configuring virtual oysters...")
oyster_params = OysterParams() # Use default biological parameters
oyster_locations = [
    (name="La Couplasse", lon=-2.0322, lat=47.0263) # 47°1'34.7"N, 2°1'55.9"W
]

virtual_oysters = VirtualOyster[]
for loc in oyster_locations
    i, j = lonlat_to_ij(grid, loc.lon, loc.lat)
    if i !== nothing && j !== nothing
        println("  -> Oyster farm '$(loc.name)' placed at grid indices (i=$i, j=$j)")
        # Place an oyster in all layers (k=grid.nz) with an initial concentration of 0.0
        for layer in 1:grid.nz
            push!(virtual_oysters, VirtualOyster(i, j, layer, oyster_params, OysterState(0.0)))
        end
    else
        println("  -> Warning: Could not find grid indices for oyster farm '$(loc.name)'.")
    end
end

oyster_tracers = (dissolved=:Virus_Dissolved, sorbed=:Virus_Sorbed)

# --- 7. Simulation and Output Parameters ---
start_time = 0.0
end_time = 2*96 * 3600.0 # Run for 12 hours
#end_time = 30*10.0 # Run for 12 hours
dt = 30.0
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]
output_directory = raw"D:\PreVir\loire_virus_sim_output"
output_interval_seconds = 60 * 60.0

# --- 8. Run the Simulation ---
println("\n--- Starting simulation ---")
println("Total duration: $(end_time / 3600.0) hours")
println("Time step (dt): $dt seconds")
println("Output will be saved to: $output_directory")

restart_file = nothing

final_state = run_simulation(
    grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
    use_adaptive_dt         = true,
    cfl_max                 = 0.8,
    dt_max                  = 120.0,
    dt_min                  = 0.01,
    dt_growth_factor        = 1.1,
    boundary_conditions     = bcs,
    sediment_params         = sediment_params,
    virtual_oysters         = virtual_oysters,
    oyster_tracers          = oyster_tracers,
    functional_interactions = functional_interactions,
    advection_scheme        = :TVD,
    D_crit                  = 0.05,
    output_dir              = output_directory,
    output_interval         = output_interval_seconds,
    restart_from            = restart_file
)

# 9. Clean Up and Summarize ---
close(ds)

println("\n--- Simulation Complete ---")
println("Final simulation time: $(round(final_state.time / 3600.0, digits=2)) hours.")
