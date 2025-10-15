# debug/debug_adsorption_exchange.jl

# --- Setup Environment ---

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.TimeSteppingModule


# --- Helper function to calculate total mass of interacting tracers ---
function calculate_total_system_mass(state::State, grid::CurvilinearGrid)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Mass of dissolved tracer in the water column
    dissolved_mass_water = 0.0
    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        dissolved_mass_water += state.tracers[:dissolved_tracer][i_glob, j_glob, k] * grid.volume[i_glob, j_glob, k]
    end

    # Mass of adsorbed tracer in the water column
    adsorbed_mass_water = 0.0
    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        adsorbed_mass_water += state.tracers[:adsorbed_tracer][i_glob, j_glob, k] * grid.volume[i_glob, j_glob, k]
    end
    
    # Mass of adsorbed tracer in the bed
    adsorbed_mass_bed = 0.0
    for j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        adsorbed_mass_bed += state.bed_mass[:adsorbed_tracer][i_glob, j_glob]
    end
    
    return dissolved_mass_water + adsorbed_mass_water + adsorbed_mass_bed
end

# --- Adsorption/Desorption Function ---
function linear_adsorption(C, env, dt)
    k_adsorption = 1e-4  # Adsorption rate (1/s)
    k_desorption = 5e-6  # Desorption rate (1/s)

    dissolved_conc = C[:dissolved_tracer]
    adsorbed_conc = C[:adsorbed_tracer]

    # Calculate mass transfer based on concentrations and rates
    adsorption_flux = k_adsorption * dissolved_conc
    desorption_flux = k_desorption * adsorbed_conc
    
    net_flux_to_adsorbed = adsorption_flux - desorption_flux
    
    # The function returns the CHANGE in concentration (dC), not the new concentration
    # dC = flux * dt
    change = net_flux_to_adsorbed * dt
    
    return Dict(:dissolved_tracer => -change, :adsorbed_tracer => +change)
end


# --- Main Debugging Logic ---

    println("--- Starting Adsorption/Sedimentation Mass Conservation Debug Script ---")

    # 1. --- Grid Setup (same as other debug script) ---
    NG = 2
    nx, ny, nz = 10, 10, 5
    Lx, Ly = 200.0, 200.0
    dx, dy = Lx / nx, Ly / ny
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    zeros_arr = zeros(nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    trues_arr_u = trues(nx_tot + 1, ny_tot)
    trues_arr_v = trues(nx_tot, ny_tot + 1)
    z_w = collect(range(0, -10, length=nz+1))
    volume = zeros(Float64, nx_tot, ny_tot, nz)
    face_area_x = zeros(Float64, nx_tot + 1, ny_tot, nz)
    face_area_y = zeros(Float64, nx_tot, ny_tot + 1, nz)
    for k in 1:nz
        dz_val = abs(z_w[k+1] - z_w[k])
        volume[:, :, k] .= dx * dy * dz_val
        face_area_x[:, :, k] .= dy * dz_val
        face_area_y[:, :, k] .= dx * dz_val
    end
    grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                        z_w, pm, pn, zeros_arr, h,
                        trues_arr_rho, trues_arr_u, trues_arr_v,
                        face_area_x, face_area_y, volume)

    # 2. --- State Setup ---
    tracer_names = (:dissolved_tracer, :adsorbed_tracer,)
    state = initialize_state(grid, tracer_names; sediment_tracers = [:adsorbed_tracer])
    state.bed_mass

    # Initial conditions: all mass is in the dissolved tracer
    fill!(state.tracers[:dissolved_tracer], 10.0)
    fill!(state.tracers[:adsorbed_tracer], 0.0)
    fill!(state.bed_mass[:adsorbed_tracer], 0.0)
    
    # Set dummy velocities
    fill!(state.u, 0.1); fill!(state.v, 0.1); fill!(state.w, 0.0)

    # 3. --- Parameters ---
    dt = 60.0 # 60 second timestep
    
    # Sedimentation parameters ONLY for the adsorbed tracer
    sediment_params = SedimentParams(ws0=1e-3, tau_cr=0.1, tau_d=0.05)
    sediment_tracers = Dict(:adsorbed_tracer => sediment_params)

    # Functional interaction for adsorption
    adsorption_interaction = FunctionalInteraction(
        affected_tracers=[:dissolved_tracer, :adsorbed_tracer],
        interaction_function=linear_adsorption
    )

    interactions = [adsorption_interaction]

    # 4. --- Pre-computation & Initial State ---
    initial_total_mass = calculate_total_system_mass(state, grid)
    using Printf
    println("\n--- Initial State ---")
    @printf "Initial Total System Mass: %.4f kg\n" initial_total_mass

    # 5. --- Run Simulation Step ---
    println("\nRunning one timestep (Reaction -> Transport)...")
    # First, the chemical reaction (source/sink terms)
    source_sink_terms!(state, grid, Vector{PointSource}(), [adsorption_interaction], 0.0, dt, 0.0)
    # Second, the vertical transport (including sedimentation of the new adsorbed tracer)
    vertical_transport!(state, grid, dt, sediment_tracers)
    
    # 6. --- Analyze Results ---
    final_total_mass = calculate_total_system_mass(state, grid)

    state.bed_mass[:adsorbed_tracer]
    state.bed_mass[:dissolved_tracer]

    state.tracers[:dissolved_tracer]

    mass_difference = final_total_mass - initial_total_mass

    println("\n--- Final State ---")
    @printf "Final Total System Mass:   %.4f kg\n" final_total_mass
    @printf "Mass Difference:           %e kg\n" mass_difference

    println("\n--- Verification ---")
    # Use a reasonable tolerance for floating point comparisons
    if abs(mass_difference) < 1e-6
        println("✅ Mass Conservation Test Passed!")
    else
        println("❌ Mass Conservation Test Failed!")
        println("  -> Mass was lost or gained during the reaction/transport step.")
    end
    
    println("\n--- Debug Script Finished ---")
