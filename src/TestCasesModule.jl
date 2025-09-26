# src/TestCases.jl

module TestCasesModule

export run_all_tests

using Test
using ..ModelStructs
using ..GridModule
using ..StateModule
using ..TimeSteppingModule
using ..SourceSinkModule
using ..HydrodynamicsModule

# --- Helper functions are unchanged ---

function center_of_mass(grid::Grid, C::Array{Float64, 3})
    total_mass = sum(C .* grid.volume)
    if total_mass == 0.0; return (0.0, 0.0, 0.0); end
    com_x = sum(grid.x .* C .* grid.volume) / total_mass
    com_y = sum(grid.y .* C .* grid.volume) / total_mass
    com_z = sum(grid.z .* C .* grid.volume) / total_mass
    return (com_x, com_y, com_z)
end

function setup_test_case()
    nx, ny, nz = 50, 50, 1
    Lx, Ly, Lz = 1000.0, 1000.0, 10.0
    grid = initialize_grid(nx, ny, nz, Lx, Ly, Lz)
    total_time = 3 * 3600.0
    dt = 30.0
    return grid, total_time, dt
end

function run_all_tests()
    @testset "HydrodynamicTransport.jl Internal Tests" begin
        
        @testset "Mass Conservation Under Dynamic Forcing" begin
            grid, total_time, dt = setup_test_case()
            state = initialize_state(grid, (:C,))
            C = state.tracers[:C]

            Lx, Ly = 1000.0, 1000.0
            center_x, center_y = Lx / 2, Ly / 2
            width = Lx / 10
            for k in 1:size(C, 3), j in 1:size(C, 2), i in 1:size(C, 1)
                x = grid.x[i,j,k]; y = grid.y[i,j,k]
                C[i,j,k] = exp(-((x - center_x)^2 / (2*width^2) + (y - center_y)^2 / (2*width^2)))
            end

            initial_mass = sum(C .* grid.volume)
            initial_com = center_of_mass(grid, C)
            
            # --- METHODERROR FIX ---
            # Call the simplified 5-argument version of the function.
            final_state = run_simulation(grid, state, 0.0, total_time, dt)
            final_C = final_state.tracers[:C]
            final_mass = sum(final_C .* grid.volume)
            final_com = center_of_mass(grid, final_C)
            
            @test isapprox(initial_mass, final_mass, rtol=1e-9)
            @test !isapprox(initial_com[1], final_com[1]) || !isapprox(initial_com[2], final_com[2])
        end

        @testset "Source/Sink Isolation" begin
            grid, _, dt = setup_test_case()
            state = initialize_state(grid, (:C_dissolved,))
            initial_concentration = 100.0
            state.tracers[:C_dissolved] .= initial_concentration
            decay_rate_per_second = 0.1 / (24 * 3600)
            expected_final_concentration = initial_concentration * (1 - decay_rate_per_second * dt)
            SourceSinkModule.source_sink_terms!(state, grid, dt)
            final_C_dissolved = state.tracers[:C_dissolved]
            @test isapprox(final_C_dissolved[1, 1, 1], expected_final_concentration, rtol=1e-9)
        end

        @testset "Hydrodynamic Forcing Update" begin
            grid, _, _ = setup_test_case()
            state = initialize_state(grid, (:C,))
            @test all(state.u .== 0.0)
            time = 0.0
            # Call the simplified placeholder function directly to test it
            HydrodynamicsModule.update_hydrodynamics!(state, time)
            @test !all(state.u .== 0.0)
            expected_u = 0.5 * cos(0)
            expected_v = 0.2 * sin(0)
            @test state.u[1,1,1] == expected_u
            @test state.v[1,1,1] == expected_v
        end

    end
end

end