# run_loire_sim.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BenchmarkTools
using Revise
using HydrodynamicTransport
using NCDatasets


# set up all data

loire_filepath = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"
hydro_data = create_hydrodynamic_data_from_file(loire_filepath)
ds = NCDataset(loire_filepath)
grid = initialize_curvilinear_grid(loire_filepath)
tracer_names = (:Virus_Dissolved, :Virus_Sorbed,)
# Specify that :Virus_Sorbed has a bed component
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list)
# A real simulation might read this from the NetCDF file if available.
state.tss .= 5.0

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

#  Define the Adsorption/Desorption Functional Interaction 
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

# Simulation and Output Parameters ---
start_time = 0.0
end_time = 30.0
dt = 6.0

bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# IMPORTANT: Update this path to your desired output location.
output_directory = raw"D:\PreVir\test_states_sorb"
output_interval_seconds = 12.0 # Save output every hour


function run_the_simulation(scheme = :ImplicitADI)
    run_simulation(
        grid, state, sources, ds, hydro_data, start_time, end_time, dt; 
        boundary_conditions = bcs,
        # --- Pass the new physics modules to the simulation ---
        sediment_tracers = sediment_params_dict,
        functional_interactions = [virus_interaction],
        # --- Other parameters ---
        advection_scheme = scheme,
        D_crit = 0.05,
        output_dir = output_directory,
        output_interval = 30.0,
        restart_from = nothing
    )
end

# warm up

run_the_simulation(:ImplicitADI)

@benchmark run_the_simulation(:ImplicitADI)

run_the_simulation(:TVD)

@benchmark run_the_simulation(:TVD)

