# test/cfl_checker_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test

# We need to explicitly import the unexported function we are testing
using HydrodynamicTransport.UtilsModule: calculate_max_cfl_term

@testset "CFL Checker Utility" begin
    println("Running unit tests for the CFL checker...")

    # --- 1. Setup a minimal test Grid ---
    ng = 2
    nx, ny, nz = 10, 10, 1
    dx, dy = 10.0, 20.0 # Use non-square cells for a robust test
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    pm = ones(nx_tot, ny_tot) ./ dx
    pn = ones(nx_tot, ny_tot) ./ dy
    grid = CurvilinearGrid(ng, nx, ny, nz,
        zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot),
        [-1.0, 0.0], pm, pn, zeros(nx_tot,ny_tot), ones(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx+1+2ng,ny+2ng), trues(nx+2ng,ny+1+2ng),
        zeros(nx_tot+1,ny_tot,nz), zeros(nx_tot,ny_tot+1,nz), fill(dx*dy, (nx_tot,ny_tot,nz)))

    @testset "Non-zero velocity field" begin
        # --- 2. Setup State with a known velocity field ---
        state = initialize_state(grid, ())
        
        # Create a simple, non-uniform velocity field
        # u increases with i, v increases with j. Max velocity will be at the top-right.
        for j in 1:ny_tot, i in 1:nx_tot+1
            state.u[i, j, 1] = (i - ng) * 0.1
        end
        for j in 1:ny_tot+1, i in 1:nx_tot
            state.v[i, j, 1] = (j - ng) * 0.05
        end

        # --- 3. Manually calculate the expected result ---
        # The maximum CFL term should be at the last physical cell (i=nx, j=ny)
        i_max, j_max = nx, ny
        i_glob, j_glob = i_max + ng, j_max + ng

        # Get the velocities around the center of the cell (i_max, j_max)
        u_face_left = state.u[i_glob, j_glob, 1]     # u at i=10
        u_face_right = state.u[i_glob+1, j_glob, 1]   # u at i=11
        v_face_bottom = state.v[i_glob, j_glob, 1]   # v at j=10
        v_face_top = state.v[i_glob, j_glob+1, 1] # v at j=11

        u_center = 0.5 * (u_face_left + u_face_right)
        v_center = 0.5 * (v_face_bottom + v_face_top)

        # The CFL term for this cell is (|u|/dx + |v|/dy)
        expected_max_cfl_term = abs(u_center) / dx + abs(v_center) / dy

        # --- 4. Call the function and verify ---
        actual_max_cfl_term = calculate_max_cfl_term(state, grid)
        
        println("Expected Max CFL Term: $expected_max_cfl_term")
        println("Actual Max CFL Term:   $actual_max_cfl_term")

        @test actual_max_cfl_term ≈ expected_max_cfl_term
    end

    @testset "Zero velocity field" begin
        # --- Setup State with a zero velocity field ---
        state_zero = initialize_state(grid, ())
        state_zero.u .= 0.0
        state_zero.v .= 0.0
        
        # The function should correctly return 0.0
        actual_max_cfl_term = calculate_max_cfl_term(state_zero, grid)
        @test actual_max_cfl_term == 0.0
    end
end