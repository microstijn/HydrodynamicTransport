# test/oyster_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test


# --- Explicitly import the function and structs we are testing ---
using HydrodynamicTransport.OysterModule: update_oysters!

@testset "Oyster Bioaccumulation Module" begin
    println("Running unit tests for the OysterModule...")

    # --- 1. Setup ---
    ng = 2; nx, ny, nz = 1, 1, 1; dx, dy, dz = 1.0, 1.0, 1.0
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    pm = ones(nx_tot, ny_tot) ./ dx; pn = ones(nx_tot, ny_tot) ./ dy
    h = ones(nx_tot, ny_tot) .* dz; z_w = [-dz, 0.0]
    volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))
    grid = CurvilinearGrid(ng, nx, ny, nz,
        zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot),
        z_w, pm, pn, zeros(nx_tot,ny_tot), h, trues(nx_tot,ny_tot), trues(nx+1+2ng,ny+2ng), trues(nx+2ng,ny+1+2ng),
        zeros(nx_tot+1,ny_tot,nz), zeros(nx_tot,ny_tot+1,nz), volume)

    dissolved_tracer = :Virus_Dissolved
    sorbed_tracer = :Virus_Sorbed
    
    oyster_params = OysterParams(
        wdw = 0.5, ϵ_free = 0.5, ϵ_sorbed = 0.1, kdep_20 = 0.23, θ_dep = 1.07
    )

    # --- Test Cases ---
    @testset "1. Filtration Rate Calculation (Indirect Test)" begin
        state_opt = initialize_state(grid, (dissolved_tracer, sorbed_tracer))
        state_opt.temperature .= 27.0; state_opt.salinity .= 15.0; state_opt.tss .= 10.0
        state_opt.tracers[dissolved_tracer] .= 100.0
        oyster_opt = VirtualOyster(1, 1, 1, oyster_params, OysterState(0.0))
        update_oysters!(state_opt, grid, [oyster_opt], 3600.0, dissolved_tracer, sorbed_tracer)

        state_temp = initialize_state(grid, (dissolved_tracer, sorbed_tracer))
        state_temp.temperature .= 10.0; state_temp.salinity .= 15.0; state_temp.tss .= 10.0
        state_temp.tracers[dissolved_tracer] .= 100.0
        oyster_temp = VirtualOyster(1, 1, 1, oyster_params, OysterState(0.0))
        update_oysters!(state_temp, grid, [oyster_temp], 3600.0, dissolved_tracer, sorbed_tracer)
        
        @test oyster_temp.state.c_oyster < oyster_opt.state.c_oyster
    end

    @testset "2. Uptake, Rejection, and Assimilation" begin
        state = initialize_state(grid, (dissolved_tracer, sorbed_tracer))
        state.temperature .= 27.0; state.salinity .= 15.0
        state.tracers[dissolved_tracer] .= 100.0
        state.tracers[sorbed_tracer] .= 50.0
        
        fr_l_day_base = 0.17 * (oyster_params.wdw^0.75) * 24.0

        # Scenario 1: Optimal TSS
        state.tss .= 10.0
        oyster_optimal = VirtualOyster(1, 1, 1, oyster_params, OysterState(0.0))
        update_oysters!(state, grid, [oyster_optimal], 86400.0, dissolved_tracer, sorbed_tracer)

        # CORRECTED: The analytical calculation now correctly includes pseudofeces rejection for TSS=10.0
        fpseudo_optimal = clamp((10.0 - oyster_params.tss_reject) / (oyster_params.tss_clog - oyster_params.tss_reject), 0.0, 1.0)
        uptake_free = fr_l_day_base * 100.0 * oyster_params.ϵ_free
        uptake_sorbed = fr_l_day_base * 50.0 * (1 - fpseudo_optimal) * oyster_params.ϵ_sorbed
        dC_dt = (uptake_free + uptake_sorbed) / oyster_params.wdw
        c_expected = 0.0 + dC_dt * 1.0
        
        @test oyster_optimal.state.c_oyster ≈ c_expected rtol=1e-5

        # Scenario 2: High TSS
        state.tss .= 60.0
        oyster_high_tss = VirtualOyster(1, 1, 1, oyster_params, OysterState(0.0))
        update_oysters!(state, grid, [oyster_high_tss], 86400.0, dissolved_tracer, sorbed_tracer)
        
        f_tss_high = 10.364 * log(60.0)^(-2.0477)
        fr_l_day_high_tss = fr_l_day_base * f_tss_high
        fpseudo_high = clamp((60.0 - oyster_params.tss_reject) / (oyster_params.tss_clog - oyster_params.tss_reject), 0.0, 1.0)

        uptake_rate_pseudo = (fr_l_day_high_tss*100.0*oyster_params.ϵ_free + fr_l_day_high_tss*50.0*(1-fpseudo_high)*oyster_params.ϵ_sorbed)
        dC_dt_pseudo = uptake_rate_pseudo / oyster_params.wdw
        c_expected_pseudo = 0.0 + dC_dt_pseudo * 1.0

        @test oyster_high_tss.state.c_oyster ≈ c_expected_pseudo rtol=1e-5
        @test oyster_high_tss.state.c_oyster < oyster_optimal.state.c_oyster
    end

    @testset "3. Depuration (Elimination)" begin
        state = initialize_state(grid, (dissolved_tracer, sorbed_tracer))
        state.temperature .= 20.0
        state.tracers[dissolved_tracer] .= 0.0
        state.tracers[sorbed_tracer] .= 0.0
        
        initial_oyster_conc = 1000.0
        oyster = VirtualOyster(1, 1, 1, oyster_params, OysterState(initial_oyster_conc))
        
        update_oysters!(state, grid, [oyster], 86400.0, dissolved_tracer, sorbed_tracer)
        
        k_dep = oyster_params.kdep_20
        expected_final_conc = initial_oyster_conc - (k_dep * initial_oyster_conc * 1.0)
        
        @test oyster.state.c_oyster ≈ expected_final_conc rtol=1e-5
    end
end