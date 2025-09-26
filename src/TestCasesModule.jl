# src/TestCases.jl

module TestCasesModule

export run_all_tests

using Test

# Import functions from our own package modules
using ..ModelStructs
using ..GridModule
using ..StateModule
using ..TimeSteppingModule

# Helper function to calculate the center of mass of a tracer field
function center_of_mass(grid, C)
    total_mass = sum(C .* grid.volume)
    return total_mass == 0 ? (0.0, 0.0) : (sum(grid.x .* C .* grid.volume) / total_mass, sum(grid.y .* C .* grid.volume) / total_mass)
end

"""
    run_all_tests()

Runs the built-in test suite for the HydrodynamicTransport package.
This function executes a series of tests to verify the correctness of the
numerical schemes, including mass conservation and advection accuracy.
"""
function run_all_tests()
    @testset "HydrodynamicTransport.jl Internal Tests" begin
        
        # --- 1. Setup a standard test case ---
        nx, ny, nz = 50, 50, 1
        Lx, Ly, Lz = 1000.0, 1000.0, 10.0
        grid = initialize_grid(nx, ny, nz, Lx, Ly, Lz)

        total_time = 400.0
        dt = 10.0

        u_velocity = 0.75 # m/s
        v_velocity = 0.5  # m/s

        state = initialize_state(grid, (:C,))
        C = state.tracers[:C]

        center_x_initial = Lx / 4
        center_y_initial = Ly / 4
        width = Lx / 10
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = grid.x[i,j,k]
            y = grid.y[i,j,k]
            C[i,j,k] = exp(-((x - center_x_initial)^2 / (2*width^2) + (y - center_y_initial)^2 / (2*width^2)))
        end

        state.u .= u_velocity
        state.v .= v_velocity
        
        initial_mass = sum(C .* grid.volume)

        # --- 2. Run the simulation ---
        final_state = run_simulation(grid, state, 0.0, total_time, dt)
        final_C = final_state.tracers[:C]

        # --- 3. Perform Tests ---
        
        @testset "Mass Conservation" begin
            final_mass = sum(final_C .* grid.volume)
            # Test that the final mass is approximately equal to the initial mass
            @test isapprox(initial_mass, final_mass, rtol=1e-6)
        end

        @testset "Center of Mass Advection" begin
            expected_x_final = center_x_initial + u_velocity * total_time
            expected_y_final = center_y_initial + v_velocity * total_time
            
            actual_x_final, actual_y_final = center_of_mass(grid, final_C)
            
            # Test that the blob moved to the correct location within a tolerance
            # of one grid cell width.
            dx = Lx / nx
            dy = Ly / ny
            @test isapprox(actual_x_final, expected_x_final, atol=dx)
            @test isapprox(actual_y_final, expected_y_final, atol=dy)
        end
    end
end

end # module TestCasesModule