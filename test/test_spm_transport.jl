# test/test_spm_transport.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Test

using Revise
using HydrodynamicTransport
using StaticArrays

# Expose the internal function for testing

using HydrodynamicTransport.VerticalTransportModule: transport_spm_column!

@testset "SPM Vertical Transport and Bed Flux" begin

    # --- 1. Minimal Setup ---
    NG = 1
    nx, ny, nz = 1, 1, 5
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
    dx, dy = 1.0, 1.0

    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    zeros_arr = zeros(nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    trues_arr_u = trues(nx_tot + 1, ny_tot)
    trues_arr_v = trues(nx_tot, ny_tot + 1)
    z_w = collect(range(0, -5, length=nz+1))
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
    state = initialize_state(grid, ()) # No generic tracers needed

    # --- Shared Parameters ---
    sediment_params = SedimentParams(
        ws0 = 0.001,
        tau_d = 0.1,
        tau_cr = 0.2,
        M = 1e-4
    )
    dt = 60.0 # 60 second timestep
    Kz = 1e-5
    g = 9.81
    i_glob, j_glob = 1 + NG, 1 + NG

    # --- 2. Deposition Test ---
    @testset "Deposition" begin
        # Reset state for this test
        state.spm .= 0.0
        state.bed_mass_per_area .= 0.0
        state.u .= 0.0
        state.v .= 0.0

        # High SPM concentration in the water column
        state.spm[i_glob, j_glob, :] .= 0.1 # 100 g/m^3
        
        # Low velocity -> low shear stress -> deposition should occur
        state.u[i_glob, j_glob, 1] = 0.01
        state.v[i_glob, j_glob, 1] = 0.01

        initial_spm_bottom = state.spm[i_glob, j_glob, 1]
        initial_bed_mass = state.bed_mass_per_area[i_glob, j_glob]

        transport_spm_column!(i_glob, j_glob, state, grid, sediment_params, Kz, g, dt)

        # Assertions
        @test state.spm[i_glob, j_glob, 1] < initial_spm_bottom
        @test state.bed_mass_per_area[i_glob, j_glob] > initial_bed_mass
    end

    # --- 3. Erosion Test ---
    @testset "Erosion" begin
        # Reset state
        state.spm .= 0.0
        state.bed_mass_per_area .= 0.0
        state.u .= 0.0
        state.v .= 0.0
        
        # High velocity -> high shear stress
        state.u[i_glob, j_glob, 1] = 1.0
        state.v[i_glob, j_glob, 1] = 1.0
        
        # Add some mass to the bed to be eroded
        state.bed_mass_per_area[i_glob, j_glob] = 0.1

        initial_spm_bottom = state.spm[i_glob, j_glob, 1]
        initial_bed_mass = state.bed_mass_per_area[i_glob, j_glob]
        
        transport_spm_column!(i_glob, j_glob, state, grid, sediment_params, Kz, g, dt)

        # Assertions
        @test state.spm[i_glob, j_glob, 1] > initial_spm_bottom
        @test state.bed_mass_per_area[i_glob, j_glob] < initial_bed_mass
    end
    
    # --- 4. No-Flux Test (Erosion limited by bed mass) ---
    @testset "No Erosion Flux" begin
        # Reset state
        state.spm .= 0.0
        state.bed_mass_per_area .= 0.0
        state.u .= 0.0
        state.v .= 0.0

        # High velocity -> high shear stress
        state.u[i_glob, j_glob, 1] = 1.0
        state.v[i_glob, j_glob, 1] = 1.0
        
        # NO mass on the bed
        state.bed_mass_per_area[i_glob, j_glob] = 0.0

        initial_spm_bottom = state.spm[i_glob, j_glob, 1]
        initial_bed_mass = state.bed_mass_per_area[i_glob, j_glob]

        transport_spm_column!(i_glob, j_glob, state, grid, sediment_params, Kz, g, dt)

        # Assertions
        @test isapprox(state.spm[i_glob, j_glob, 1], initial_spm_bottom, atol=1e-9)
        @test isapprox(state.bed_mass_per_area[i_glob, j_glob], initial_bed_mass, atol=1e-9)
    end

end