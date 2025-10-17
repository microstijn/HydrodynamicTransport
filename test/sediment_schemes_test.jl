# test/sediment_schemes_test.jl
using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using Test
# Explicitly import the internal functions we are testing
using HydrodynamicTransport.VerticalTransportModule: _apply_sedimentation_forward_euler!, _apply_sedimentation_backward_euler!, apply_sedimentation!

println("--- Unit Tests for Sedimentation Schemes (Forward vs. Backward Euler) ---")
# test/sediment_schemes_test.jl

println("--- Unit Tests for Sedimentation Schemes (Forward vs. Backward Euler) ---")

@testset "Sedimentation Scheme Comparison" begin

    # --- 1. Common Setup ---
    NG = 1
    nx, ny, nz = 1, 1, 1
    dx, dy, dz = 10.0, 10.0, 1.0
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
    
    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)
    face_area_x = ones(Float64, nx_tot+1, ny_tot, nz) .* (dy*dz)
    face_area_y = ones(Float64, nx_tot, ny_tot+1, nz) .* (dx*dz)
    z_w = [-dz, 0.0]
    
    grid = CurvilinearGrid(NG, nx, ny, nz, zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), 
                           z_w, pm, pn, zeros(nx_tot,ny_tot), h,
                           trues(nx_tot,ny_tot), trues(nx_tot+1,ny_tot), trues(nx_tot,ny_tot+1),
                           face_area_x, face_area_y, volume)

    sed_params_explicit = SedimentParams(
        ws0 = 0.001, tau_d = 0.1, tau_cr = 0.2, M = 1e-4, settling_scheme = :ForwardEuler
    )
    sed_params_implicit = SedimentParams(
        ws0 = 0.001, tau_d = 0.1, tau_cr = 0.2, M = 1e-4, settling_scheme = :BackwardEuler
    )
    
    dt = 60.0; g = 9.81; D_crit = 0.05
    i_glob, j_glob = 1 + NG, 1 + NG
    cell_volume = volume[i_glob, j_glob, 1]

    # --- 2. Test Cases ---
    
    @testset "Backward Euler: Pure Deposition & Mass Conservation" begin
        state = initialize_state(grid, ())
        state.u[i_glob, j_glob, 1] = 0.01
        state.v[i_glob, j_glob, 1] = 0.01

        C_col = [10.0]
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1.0

        initial_mass_in_water = C_col[1] * cell_volume
        initial_mass_in_bed = bed_mass[i_glob, j_glob]
        initial_total_mass = initial_mass_in_water + initial_mass_in_bed

        _apply_sedimentation_backward_euler!(C_col, bed_mass, grid, state, sed_params_implicit, i_glob, j_glob, dt, g, D_crit)

        final_mass_in_water = C_col[1] * cell_volume
        final_mass_in_bed = bed_mass[i_glob, j_glob]
        final_total_mass = final_mass_in_water + final_mass_in_bed
        
        @test C_col[1] < 10.0
        @test bed_mass[i_glob, j_glob] > 1.0
        @test isapprox(initial_total_mass, final_total_mass, rtol=1e-12)
    end

    @testset "Backward Euler: Pure Resuspension & Mass Conservation" begin
        state = initialize_state(grid, ())
        state.u[i_glob, j_glob, 1] = 1.0
        state.v[i_glob, j_glob, 1] = 1.0

        C_col = [0.0]
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1000.0

        initial_mass_in_water = C_col[1] * cell_volume
        initial_mass_in_bed = bed_mass[i_glob, j_glob]
        initial_total_mass = initial_mass_in_water + initial_mass_in_bed

        _apply_sedimentation_backward_euler!(C_col, bed_mass, grid, state, sed_params_implicit, i_glob, j_glob, dt, g, D_crit)

        final_mass_in_water = C_col[1] * cell_volume
        final_mass_in_bed = bed_mass[i_glob, j_glob]
        final_total_mass = final_mass_in_water + final_mass_in_bed

        @test C_col[1] > 0.0
        @test bed_mass[i_glob, j_glob] < 1000.0
        @test isapprox(initial_total_mass, final_total_mass, rtol=1e-12)
    end
    
    @testset "D_crit check prevents flux" begin
        state = initialize_state(grid, ())
        state.u[i_glob, j_glob, 1] = 1.0 # High velocity (for erosion)

        # Make the cell depth shallower than D_crit
        shallow_grid = deepcopy(grid)
        shallow_grid.volume[i_glob, j_glob, 1] = 0.01 * (dx * dy) # dz = 0.01
        
        C_col = [10.0] # High concentration
        bed_mass = zeros(nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1000.0 # High bed mass

        initial_C = C_col[1]
        initial_bed_mass = bed_mass[i_glob, j_glob]
        
        # Call with D_crit = 0.05, which is > dz
        _apply_sedimentation_backward_euler!(C_col, bed_mass, shallow_grid, state, sed_params_implicit, i_glob, j_glob, dt, g, 0.05)

        # No change should occur
        @test C_col[1] == initial_C
        @test bed_mass[i_glob, j_glob] == initial_bed_mass
    end

    @testset "Dispatcher correctly calls Forward Euler" begin
        state = initialize_state(grid, ())
        state.u[i_glob, j_glob, 1] = 0.01
        
        C_col_fe = [10.0]; bed_mass_fe = zeros(nx_tot, ny_tot)
        C_col_dispatch = [10.0]; bed_mass_dispatch = zeros(nx_tot, ny_tot)

        _apply_sedimentation_forward_euler!(C_col_fe, bed_mass_fe, grid, state, sed_params_explicit, i_glob, j_glob, dt, g, D_crit)
        apply_sedimentation!(C_col_dispatch, bed_mass_dispatch, grid, state, sed_params_explicit, i_glob, j_glob, dt, g, D_crit)
        
        @test C_col_fe[1] == C_col_dispatch[1]
        @test bed_mass_fe[i_glob, j_glob] == bed_mass_dispatch[i_glob, j_glob]
    end

    @testset "Dispatcher correctly calls Backward Euler" begin
        state = initialize_state(grid, ())
        state.u[i_glob, j_glob, 1] = 0.01
        
        C_col_be = [10.0]; bed_mass_be = zeros(nx_tot, ny_tot)
        C_col_dispatch = [10.0]; bed_mass_dispatch = zeros(nx_tot, ny_tot)

        _apply_sedimentation_backward_euler!(C_col_be, bed_mass_be, grid, state, sed_params_implicit, i_glob, j_glob, dt, g, D_crit)
        apply_sedimentation!(C_col_dispatch, bed_mass_dispatch, grid, state, sed_params_implicit, i_glob, j_glob, dt, g, D_crit)
        
        @test C_col_be[1] == C_col_dispatch[1]
        @test bed_mass_be[i_glob, j_glob] == bed_mass_dispatch[i_glob, j_glob]
    end
    
    @testset "Dispatcher throws error for unknown scheme" begin
        state = initialize_state(grid, ())
        C_col = [10.0]; bed_mass = zeros(nx_tot, ny_tot)
        bad_params = SedimentParams(settling_scheme = :UnknownScheme)
        
        @test_throws ErrorException apply_sedimentation!(C_col, bed_mass, grid, state, bad_params, i_glob, j_glob, dt, g, D_crit)
    end

end

println("\n✅ All new sedimentation scheme unit tests passed.")