# run_vortex_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test
using UnicodePlots
using ProgressMeter
using Base: @kwdef # Needed for the local struct definition

# --- Explicitly import unexported functions needed for the custom loop ---
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.SourceSinkModule: source_sink_terms!
using HydrodynamicTransport.SettlingModule: apply_settling!
using HydrodynamicTransport.BedExchangeModule: bed_exchange!

# --- NEW: Local definitions for Decay physics, as they are not yet in the main package ---
@kwdef struct DecayParams
    tracer_name::Symbol
    base_rate::Float64 = 0.0
    temp_ref::Float64 = 20.0
    temp_theta::Float64 = 1.0
    light_coeff::Float64 = 0.0
end

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
# ------------------------------------------------------------------------------------

println("--- HydrodynamicTransport.jl: Full Integration Test with Vortex Dynamics ---")

# --- 1. Custom Curvilinear Grid Generation ---
println("Generating custom 50x50x5 curvilinear grid...")
ng = 2
nx, ny, nz = 50, 50, 5
Lx, Ly, Lz = 1000.0, 1000.0, 20.0
dx, dy = Lx / nx, Ly / ny
nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

pm = ones(Float64, nx_tot, ny_tot) ./ dx
pn = ones(Float64, nx_tot, ny_tot) ./ dy
h = ones(Float64, nx_tot, ny_tot) .* Lz
zeros_arr = zeros(Float64, nx_tot, ny_tot)
trues_arr_rho = trues(nx_tot, ny_tot)
trues_arr_u = trues(nx + 1 + 2*ng, ny + 2*ng)
trues_arr_v = trues(nx + 2*ng, ny + 1 + 2*ng)
z_w = collect(range(-Lz, 0, length=nz+1))

volume = zeros(Float64, nx_tot, ny_tot, nz)
face_area_x = zeros(Float64, nx_tot + 1, ny_tot, nz)
face_area_y = zeros(Float64, nx_tot, ny_tot + 1, nz)

for k in 1:nz
    dz = abs(z_w[k+1] - z_w[k])
    volume[:, :, k] .= dx * dy * dz
    face_area_x[:, :, k] .= dy * dz
    face_area_y[:, :, k] .= dx * dz
end

grid = CurvilinearGrid(ng, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                       z_w, pm, pn, zeros_arr, h,
                       trues_arr_rho, trues_arr_u, trues_arr_v,
                       face_area_x, face_area_y, volume)

# --- 2. Custom Hydrodynamics: Steady-State Vortex ---
println("Generating steady-state vortex flow field...")
function generate_vortex_flow!(state::State, grid::CurvilinearGrid)
    center_x, center_y = grid.nx * dx / 2.0, grid.ny * dy / 2.0
    omega = 0.001
    for k in 1:grid.nz, j in 1:grid.ny, i in 1:(grid.nx + 1)
        x_face = (i - 0.5) * dx
        y_center = (j - 0.5) * dy
        state.u[i+ng, j+ng, k] = -omega * (y_center - center_y)
    end
    for k in 1:grid.nz, j in 1:(grid.ny + 1), i in 1:grid.nx
        x_center = (i - 0.5) * dx
        y_face = (j - 0.5) * dy
        state.v[i+ng, j+ng, k] = omega * (x_center - center_x)
    end
end

# --- 3. State Initialization ---
println("Initializing state with three interacting tracers...")
tracer_names = (:Dye, :Reagent, :Sediment)
sediment_tracer_list = [:Sediment]
state = initialize_state(grid, tracer_names; sediment_tracers=sediment_tracer_list)

# --- 4. Physics Configuration ---
println("Configuring sources, interactions, and sediment parameters...")
sources = [
    PointSource(i=10, j=25, k=nz, tracer_name=:Dye, influx_rate=(t)->1.0e4),
    PointSource(i=22, j=25, k=nz, tracer_name=:Reagent, influx_rate=(t)->1.0e4)
]

sediment_params = Dict(
    :Sediment => SedimentParams(ws = 0.002, erosion_rate = 1e-8)
)

decay_interaction = create_decay_interaction(DecayParams(
    tracer_name = :Reagent,
    base_rate = 1.0 / (2 * 3600.0) # 2-hour half-life
))

function dye_reacts_with_reagent(concentrations, environment, dt)
    k_react = 1e-3
    C_dye = max(0.0, concentrations[:Dye])
    C_reagent = max(0.0, concentrations[:Reagent])
    delta_C = min(C_dye, C_reagent, k_react * C_dye * C_reagent * dt)
    return Dict(:Dye => -delta_C, :Reagent => -delta_C, :Sediment => +delta_C)
end

reaction_interaction = FunctionalInteraction(
    affected_tracers = [:Dye, :Reagent, :Sediment],
    interaction_function = dye_reacts_with_reagent
)
functional_interactions = [decay_interaction, reaction_interaction]

# --- 5. Custom Simulation Loop ---
println("Starting custom simulation loop...")
start_time = 0.0
dt = 14.0
end_time = 6 * 3600.0
time_range = start_time:dt:end_time
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

initial_reagent_mass = sum(state.tracers[:Reagent] .* grid.volume)
total_source_mass_added = 0.0


@showprogress "Simulating..." for time in time_range
    global total_source_mass_added
    if time == start_time; continue; end
    state.time = time

    apply_boundary_conditions!(state, grid, bcs)
    generate_vortex_flow!(state, grid)
    horizontal_transport!(state, grid, dt, :TVD, 0.0, bcs)
    vertical_transport!(state, grid, dt)
    deposition = apply_settling!(state, grid, dt, sediment_params)
    bed_exchange!(state, grid, dt, deposition, sediment_params)
    source_sink_terms!(state, grid, sources, functional_interactions, time, dt, 0.0)

    total_source_mass_added += (sources[1].influx_rate(time) + sources[2].influx_rate(time)) * dt
end

println("\n--- Simulation Complete ---")

# --- 6. Verification and Visualization ---
println("Running verification tests...")
@testset "Vortex Simulation Verification" begin
    final_sediment_water = sum(state.tracers[:Sediment] .* grid.volume)
    final_sediment_bed = sum(state.bed_mass[:Sediment]) * (dx*dy)
    @test final_sediment_water > 0
    @test final_sediment_bed > 0
    final_reagent_mass = sum(state.tracers[:Reagent] .* grid.volume)
    @test final_reagent_mass < initial_reagent_mass + total_source_mass_added
    total_final_mass = sum(sum(t .* grid.volume) for t in values(state.tracers)) + sum(sum(b) for b in values(state.bed_mass)) * (dx*dy)
    @test total_final_mass <= total_source_mass_added + 1e-9
end

println("\n--- Final Tracer Distributions (Surface Layer) ---")

function plot_tracer(tracer_array, title)
    surface_data = tracer_array[ng+1:nx+ng, ng+1:ny+ng, nz]
    println(heatmap(surface_data', title=title, colormap=:viridis, width=50))
end

plot_tracer(state.tracers[:Dye], "Final Dye Concentration")
plot_tracer(state.tracers[:Reagent], "Final Reagent Concentration")
plot_tracer(state.tracers[:Sediment], "Final Sediment Concentration (in water)")

println("\n--- Test script finished successfully! ---")