# run_loire_simulation.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using HydrodynamicTransport.FluxLimitersModule
using NCDatasets
import Proj

# Data Configuration 
loire_filepath = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"

hydro_data = create_hydrodynamic_data_from_file(loire_filepath);

# Grid and State Initialization
ds = NCDataset(loire_filepath);
grid = initialize_curvilinear_grid(loire_filepath);
tracer_names = (:Virus_Dissolved, :Virus_Sorbed)
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list);

# Set a uniform background TSS concentration (e.g., 10.0 mg/L, which is g/m^3)
# A real simulation might read this from the NetCDF file if available.
state.tss .= 10.0
# Also set uniform Temp and Salinity for the oyster model if they are not in the hydro_data
if !haskey(hydro_data.var_map, :temp); state.temperature .= 15.0; end
if !haskey(hydro_data.var_map, :salt); state.salinity .= 25.0; end

# Source Configuration
sources = PointSource[]
source_locations = [
    (name = "Nantes",        lon = -1.549464,  lat = 47.197319),
    (name = "Saint-Nazaire", lon = -2.28,      lat = 47.27),
    (name = "Cordemais",     lon = -1.97,      lat = 47.28)
]

begin
    target_crs = "EPSG:4326"
    source_crs = "EPSG:2154" # its a French speciality 

    trans = Proj.Transformation(source_crs, target_crs)

    x_ys = [
        (309300, 6702200), 
        (322054, 6707309),
        (310326, 6692120),
        (355092, 6688872),
        (350512.47, 6688894.28)
    ]

    lat_lons = [trans((xy[1], xy[2])) for xy in x_ys]
    new_data = [(name = "Point $i", lon = lon, lat = lat) for (i, (lat, lon)) in enumerate(lat_lons)]
    append!(source_locations, new_data)
end

for loc in source_locations
    i, j = lonlat_to_ij(grid, loc.lon, loc.lat)
    if i !== nothing && j !== nothing
        println("  -> Source '$(loc.name)' placed at grid indices (i=$i, j=$j)")
        push!(sources, PointSource(i=i, j=j, k=grid.nz, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry=true))
    else
        println("  -> Warning: Could not find grid indices for source '$(loc.name)'.")
    end
end

# Define Sediment Parameters for the Sorbed Tracer 
sediment_params = Dict(
    :Virus_Sorbed => SedimentParams(ws = 0.0005, erosion_rate = 1e-7, tau_ce = 0.1)
)

# Define the Adsorption/Desorption Functional Interaction 
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

# Oyster Configuration 
oyster_params = OysterParams() # Use default biological parameters
oyster_locations = [
    (name = "La Couplasse", lon=-2.0322, lat=47.0263),
    (name = "Plage de Villès-Martin",  lon = -2.225092, lat = 47.257386),
    (name = "Phare à terre de Villès-Martin",  lon = -2.227273, lat = 47.255451),
    (name = "Villès-Martin L'embouchure de la Loire",  lon = -2.223111, lat = 47.259898),
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

# Simulation and Output Parameters 
start_time = 6 * 3600.0
end_time = 120 * 3600.0 # Run for 12 hours
#end_time = 30*10.0 # Run for 12 hours
dt = 20.0
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]
output_directory = raw"D:\PreVir\loire_virus_sim_outputADi"
output_interval_seconds = 60 * 60.0

# Run the Simulation
restart_file = nothing

final_state = run_simulation(
    grid, state, sources, start_time, end_time, dt; 
    ds,
    hydro_data, 
    use_adaptive_dt         = true,
    cfl_max                 = 0.9,
    dt_max                  = 1500.0,
    dt_min                  = 0.01,
    dt_growth_factor        = 1.1,
    boundary_conditions     = bcs,
    sediment_params         = sediment_params,
    virtual_oysters         = virtual_oysters,
    oyster_tracers          = oyster_tracers,
    functional_interactions = functional_interactions,
    advection_scheme        = :TVD,
    #limiter_func            = van_leer,      
    D_crit                  = 0.05,
    output_dir              = output_directory,
    output_interval         = output_interval_seconds,
    restart_from            = restart_file
)

sum(final_state.tracers[:Virus_Dissolved][:, :, 1])
minimum(final_state.tracers[:Virus_Dissolved][:, :, 1])
maximum(final_state.tracers[:Virus_Dissolved][:, :, 1])
sum(final_state.tracers[:Virus_Sorbed][:, :, 1])
sum(final_state.bed_mass[:Virus_Sorbed])

# Clean
close(ds)

println("\n--- Simulation Complete ---")
println("Final simulation time: $(round(final_state.time / 3600.0, digits=2)) hours.")

