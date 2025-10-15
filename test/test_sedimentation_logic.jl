# test/test_sedimentation_logic.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using HydrodynamicTransport

# We need to explicitly import the internal function we are testing
using HydrodynamicTransport.VerticalTransportModule: _apply_sedimentation!

println("--- Unit Test for Sedimentation Logic (_apply_sedimentation!) ---")

@testset "Sedimentation Bed Flux Logic" begin

    # --- 1. Minimal Setup for a Single Column ---
    NG = 1
    nx, ny, nz = 1, 1, 1 # A single cell column is sufficient
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

    # Sediment parameters for testing
    sed_params = SedimentParams(
        ws0 = 0.001,      # 1 mm/s settling velocity
        tau_d = 0.1,      # Deposition starts below this shear stress
        tau_cr = 0.2,     # Erosion starts above this shear stress
        M = 1e-4          # Erosion rate parameter
    )
    dt = 60.0; g = 9.81
    i_glob, j_glob = 1 + NG, 1 + NG

    @testset "Case 1: Pure Deposition" begin
        state = initialize_state(grid, ())
        # Low velocity -> low shear stress
        state.u[i_glob, j_glob, 1] = 0.01 
        state.v[i_glob, j_glob, 1] = 0.01

        C_col = [10.0] # High concentration in water (10 g/m^3)
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1.0 # 1g of mass on the bed

        initial_C = C_col[1]
        initial_bed_mass = bed_mass[i_glob, j_glob]

        _apply_sedimentation!(C_col, bed_mass, grid, state, sed_params, i_glob, j_glob, dt, g)

        @test C_col[1] < initial_C
        @test bed_mass[i_glob, j_glob] > initial_bed_mass
        @test !any(isnan, C_col) && !any(isnan, bed_mass)
    end

    @testset "Case 2: Pure Erosion" begin
        state = initialize_state(grid, ())
        # High velocity -> high shear stress
        state.u[i_glob, j_glob, 1] = 1.0
        state.v[i_glob, j_glob, 1] = 1.0

        C_col = [0.0] # No concentration in water
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1000.0 # High mass on the bed to erode

        initial_C = C_col[1]
        initial_bed_mass = bed_mass[i_glob, j_glob]
        
        _apply_sedimentation!(C_col, bed_mass, grid, state, sed_params, i_glob, j_glob, dt, g)

        @test C_col[1] > initial_C
        @test bed_mass[i_glob, j_glob] < initial_bed_mass
        @test !any(isnan, C_col) && !any(isnan, bed_mass)
    end

    @testset "Case 3: Erosion Limited by Bed Mass (Stability Check)" begin
        state = initialize_state(grid, ())
        # High velocity -> high shear stress
        state.u[i_glob, j_glob, 1] = 1.0
        state.v[i_glob, j_glob, 1] = 1.0

        C_col = [0.0]
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 0.0 # NO mass on the bed

        _apply_sedimentation!(C_col, bed_mass, grid, state, sed_params, i_glob, j_glob, dt, g)

        # No mass should be created. Concentration and bed mass should remain unchanged.
        @test C_col[1] == 0.0
        @test bed_mass[i_glob, j_glob] == 0.0
        @test !any(isnan, C_col) && !any(isnan, bed_mass)
    end

    @testset "Case 4: Deposition Limited by Water Concentration (Stability Check)" begin
        state = initialize_state(grid, ())
        # Low velocity
        state.u[i_glob, j_glob, 1] = 0.001
        state.v[i_glob, j_glob, 1] = 0.001
        
        # Use a very high settling velocity and dt to force the clamp
        fast_sed_params = SedimentParams(ws0 = 10.0, tau_d = 0.1, tau_cr = 0.2)
        fast_dt = 1.0
        
        C_col = [10.0]
        bed_mass = zeros(Float64, nx_tot, ny_tot)

        initial_total_mass = (C_col[1] * volume[i_glob, j_glob, 1]) + bed_mass[i_glob, j_glob]
        
        _apply_sedimentation!(C_col, bed_mass, grid, state, fast_sed_params, i_glob, j_glob, fast_dt, g)

        final_total_mass = (C_col[1] * volume[i_glob, j_glob, 1]) + bed_mass[i_glob, j_glob]

        # The concentration should not become negative
        @test C_col[1] >= 0.0
        # The final mass must be conserved
        @test isapprox(initial_total_mass, final_total_mass)
        @test !any(isnan, C_col) && !any(isnan, bed_mass)
    end
    
    @testset "Case 5: Zero Volume Cell (Division by Zero Check)" begin
        # Create a grid with a zero-volume cell
        zero_vol_grid = deepcopy(grid)
        zero_vol_grid.volume[i_glob, j_glob, 1] = 0.0

        state = initialize_state(zero_vol_grid, ())
        state.u[i_glob, j_glob, 1] = 0.01 
        
        C_col = [10.0]
        bed_mass = zeros(Float64, nx_tot, ny_tot)
        bed_mass[i_glob, j_glob] = 1.0
        
        # This function call should not throw a division-by-zero error or create NaNs
        _apply_sedimentation!(C_col, bed_mass, zero_vol_grid, state, sed_params, i_glob, j_glob, dt, g)

        # Because the volume is zero, no change should have occurred
        @test C_col[1] == 10.0
        @test bed_mass[i_glob, j_glob] == 1.0
        @test !any(isnan, C_col) && !any(isnan, bed_mass)
    end

end

println("\n✅ All sedimentation logic unit tests passed.")