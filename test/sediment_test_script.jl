# test/sediment_test_script.jl

# This script provides an integration test for the sediment transport model,
# including bed flux, nudging, and interaction with other tracers, using a
# CurvilinearGrid.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test


@testset "Sediment Flag Integration Test" begin
    # --- 1. Grid and State Setup ---
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
        dz = abs(z_w[k+1] - z_w[k])
        volume[:, :, k] .= dx * dy * dz
        face_area_x[:, :, k] .= dy * dz
        face_area_y[:, :, k] .= dx * dz
    end

    grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                           z_w, pm, pn, zeros_arr, h,
                           trues_arr_rho, trues_arr_u, trues_arr_v,
                           face_area_x, face_area_y, volume)

    # Define all tracers for the simulation
    tracer_names = (:my_sediment, :virus_dissolved, :virus_particulate)
    # Define which of the above tracers should have a corresponding bed_mass array
    sediment_tracer_list = [:my_sediment]
    # Initialize the state using the new keyword argument
    state = initialize_state(grid, tracer_names; sediment_tracers=sediment_tracer_list)


    # --- 2. Initial Conditions and Parameters ---
    # Set initial concentration for our sediment tracer in the physical domain only
    C_phys_view = view(state.tracers[:my_sediment], (NG+1):(nx+NG), (NG+1):(ny+NG), :)
    C_phys_view .= 0.1

    # Set a non-uniform TSS field (which is NOT the settling tracer)
    for i in 1:nx_tot, j in 1:ny_tot
        i_phys = max(1, i - NG)
        state.tss[i, j, :] .= 1.0 + 0.5 * sin(2 * pi * i_phys / nx)
    end
    state.tracers[:virus_dissolved][(4+NG):(6+NG), (4+NG):(6+NG), 1] .= 100.0

    # Define the sediment parameter dictionary
    sediment_tracers_dict = Dict(
        :my_sediment => SedimentParams(ws0 = 0.002)
    )

    # --- 3. Adsorption/Desorption Interaction ---
    function virus_adsorption(C, env, dt)
        Kd = 0.1
        C_dissolved = max(0.0, C[:virus_dissolved])
        C_particulate = max(0.0, C[:virus_particulate])
        C_tss = env.TSS
        
        C_particulate_eq = Kd * C_tss * C_dissolved
        transfer_rate = 0.001
        delta_C = (C_particulate_eq - C_particulate) * transfer_rate * dt
        
        if delta_C > 0; delta_C = min(delta_C, C_dissolved); else; delta_C = max(delta_C, -C_particulate); end
        
        return Dict(:virus_dissolved => -delta_C, :virus_particulate => +delta_C)
    end
    adsorption_interaction = FunctionalInteraction(
        affected_tracers = [:virus_dissolved, :virus_particulate],
        interaction_function = virus_adsorption
    )

    # --- 4. Simulation Execution ---
    start_time = 0.0
    end_time = 3600.0 * 2
    dt = 10.0
    
    final_state = run_simulation(
        grid, state, Vector{PointSource}(), start_time, end_time, dt;
        sediment_tracers = sediment_tracers_dict,
        functional_interactions = [adsorption_interaction]
    )

    # --- 5. Results Verification ---
    @testset "Bed Deposition Verification" begin
        # Check that the settling tracer has deposited mass to the bed
        @test haskey(final_state.bed_mass, :my_sediment)
        # Sum only over physical domain
        total_bed_mass = sum(view(final_state.bed_mass[:my_sediment], (NG+1):(nx+NG), (NG+1):(ny+NG)))
        @test total_bed_mass > 0.0
    end

    # --- CORRECTED MASS CONSERVATION TEST ---
    @testset "Mass Conservation" begin
        # Get views of the physical domain to correctly calculate mass
        ng = grid.ng
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        
        phys_domain_3d = ((ng+1):(nx+ng), (ng+1):(ny+ng), 1:nz)
        phys_domain_2d = ((ng+1):(nx+ng), (ng+1):(ny+ng))
        volume_phys = view(grid.volume, phys_domain_3d...)

        # Correctly calculate initial mass (Concentration * Volume)
        initial_water_mass = sum(view(state.tracers[:my_sediment], phys_domain_3d...) .* volume_phys)
        initial_bed_mass = sum(view(state.bed_mass[:my_sediment], phys_domain_2d...))
        total_initial_mass = initial_water_mass + initial_bed_mass

        # Correctly calculate final mass
        final_water_mass = sum(view(final_state.tracers[:my_sediment], phys_domain_3d...) .* volume_phys)
        final_bed_mass = sum(view(final_state.bed_mass[:my_sediment], phys_domain_2d...))
        total_final_mass = final_water_mass + final_bed_mass
        
        # Check that total mass of the settling tracer is conserved (water + bed)
        @test isapprox(total_initial_mass, total_final_mass, rtol=1e-6)

        # Check that total virus mass is conserved (since it does not settle in this model)
        initial_total_virus = sum(view(state.tracers[:virus_dissolved], phys_domain_3d...) .* volume_phys) + 
                              sum(view(state.tracers[:virus_particulate], phys_domain_3d...) .* volume_phys)
        final_total_virus = sum(view(final_state.tracers[:virus_dissolved], phys_domain_3d...) .* volume_phys) + 
                            sum(view(final_state.tracers[:virus_particulate], phys_domain_3d...) .* volume_phys)
        @test isapprox(initial_total_virus, final_total_virus, rtol=1e-6)
    end
end