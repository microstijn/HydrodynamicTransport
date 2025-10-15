# test/test_nudging.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# test/test_nudging.jl

using Test
using HydrodynamicTransport

# Expose the internal function for testing
using HydrodynamicTransport.SourceSinkModule: nudge_spm_towards_tss!

@testset "SPM to TSS Nudging" begin

    # --- 1. Minimal Setup ---
    NG = 1
    nx, ny, nz = 1, 1, 1
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
    dx, dy = 1.0, 1.0

    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    zeros_arr = zeros(nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    trues_arr_u = trues(nx_tot + 1, ny_tot)
    trues_arr_v = trues(nx_tot, ny_tot + 1)
    z_w = collect(range(0, -1, length=nz+1))
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
    state = initialize_state(grid, ())
    dt = 60.0 # 60 second timestep

    # --- 2. Nudging Up Test ---
    @testset "Nudging Up" begin
        state.spm[1+NG, 1+NG, 1] = 0.1
        state.tss[1+NG, 1+NG, 1] = 1.0
        
        # Nudge over 1 hour (3600s)
        params = SedimentParams(spm_nudging_timescale = 3600.0)
        
        initial_spm = state.spm[1+NG, 1+NG, 1]
        nudge_spm_towards_tss!(state, params, dt)
        
        @test state.spm[1+NG, 1+NG, 1] > initial_spm
        @test state.spm[1+NG, 1+NG, 1] < state.tss[1+NG, 1+NG, 1]
    end

    # --- 3. Nudging Down Test ---
    @testset "Nudging Down" begin
        state.spm[1+NG, 1+NG, 1] = 1.0
        state.tss[1+NG, 1+NG, 1] = 0.1
        
        params = SedimentParams(spm_nudging_timescale = 3600.0)
        
        initial_spm = state.spm[1+NG, 1+NG, 1]
        nudge_spm_towards_tss!(state, params, dt)
        
        @test state.spm[1+NG, 1+NG, 1] < initial_spm
        @test state.spm[1+NG, 1+NG, 1] > state.tss[1+NG, 1+NG, 1]
    end

    # --- 4. No Nudging Test ---
    @testset "No Nudging" begin
        state.spm[1+NG, 1+NG, 1] = 0.5
        state.tss[1+NG, 1+NG, 1] = 1.0
        
        # Timescale is nothing, so no nudging should occur
        params = SedimentParams(spm_nudging_timescale = nothing)
        
        initial_spm = state.spm[1+NG, 1+NG, 1]
        nudge_spm_towards_tss!(state, params, dt)
        
        @test state.spm[1+NG, 1+NG, 1] == initial_spm
    end

    # --- 5. Full Nudging Test ---
    @testset "Full Nudging" begin
        state.spm[1+NG, 1+NG, 1] = 0.1
        state.tss[1+NG, 1+NG, 1] = 1.0
        
        # Nudging timescale is equal to the timestep, should force spm to equal tss
        params = SedimentParams(spm_nudging_timescale = dt)
        
        nudge_spm_towards_tss!(state, params, dt)
        
        @test isapprox(state.spm[1+NG, 1+NG, 1], state.tss[1+NG, 1+NG, 1])
    end

end