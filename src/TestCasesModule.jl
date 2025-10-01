# src/TestCasesModule.jl

module TestCasesModule

export run_all_tests

using Test
using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.GridModule
using ..HydrodynamicTransport.StateModule
using ..HydrodynamicTransport.HorizontalTransportModule
using ..HydrodynamicTransport.VerticalTransportModule
using ..HydrodynamicTransport.SourceSinkModule
using ..HydrodynamicTransport.BoundaryConditionsModule 
using ..HydrodynamicTransport.VectorOperationsModule
using ..HydrodynamicTransport.HydrodynamicsModule
using StaticArrays

# --- Test Suite Setup ---
const NG = 2 # Use a constant for the number of ghost cells in tests

function setup_cartesian_test(;nx=20, ny=20, nz=5)
    Lx, Ly, Lz = 100.0, 100.0, 10.0
    # Initialize grid with ghost cells
    grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz; ng=NG)
    state = initialize_state(grid, (:C,))
    return grid, state
end

# --- Main Test Runner ---
function run_all_tests()
    @testset "All Unit Tests" begin
        
        @testset "Grid and State Initialization with Ghost Cells" begin
            nx, ny, nz = 20, 20, 5
            grid, state = setup_cartesian_test(nx=nx, ny=ny, nz=nz)

            @test grid.ng == NG
            @test grid.dims == SVector(nx, ny, nz)
            
            # Check that tracer arrays have the full dimensions
            nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
            @test size(state.tracers[:C]) == (nx_tot, ny_tot, nz)
            
            # Check staggered arrays
            @test size(state.u) == (nx_tot + 1, ny_tot, nz)
            @test size(state.v) == (nx_tot, ny_tot + 1, nz)
        end

        @testset "Boundary Conditions Module" begin
            @testset "OpenBoundary correctly extrapolates" begin
                nx, ny = 10, 10
                grid, state = setup_cartesian_test(nx=nx, ny=ny, nz=1)
                C = state.tracers[:C]
                
                # --- FIX: Set values unambiguously to avoid overwriting corners ---
                # Set side values, excluding corners
                C[NG+1,  NG+2:ny+NG-1, 1] .= 10.0 # West side
                C[nx+NG, NG+2:ny+NG-1, 1] .= 20.0 # East side
                C[NG+2:nx+NG-1, NG+1,  1] .= 30.0 # South side
                C[NG+2:nx+NG-1, ny+NG, 1] .= 40.0 # North side

                # Set corner values explicitly
                C[NG+1, NG+1, 1] = 11.0 # SW corner
                C[nx+NG, NG+1, 1] = 21.0 # SE corner
                C[NG+1, ny+NG, 1] = 14.0 # NW corner
                C[nx+NG, ny+NG, 1] = 24.0 # NE corner

                bcs = [
                    OpenBoundary(side=:West), OpenBoundary(side=:East),
                    OpenBoundary(side=:South), OpenBoundary(side=:North)
                ]
                apply_boundary_conditions!(state, grid, bcs)

                # --- FIX: More precise tests for sides and corners ---
                # Test sides (excluding corners)
                @test all(C[1:NG, NG+2:ny+NG-1, 1] .== 10.0)
                @test all(C[nx+NG+1:end, NG+2:ny+NG-1, 1] .== 20.0)
                @test all(C[NG+2:nx+NG-1, 1:NG, 1] .== 30.0)
                @test all(C[NG+2:nx+NG-1, ny+NG+1:end, 1] .== 40.0)
                
                # Test corners explicitly
                @test all(C[1:NG, 1:NG, 1] .== 11.0) # SW
                @test all(C[nx+NG+1:end, 1:NG, 1] .== 21.0) # SE
                @test all(C[1:NG, ny+NG+1:end, 1] .== 14.0) # NW
                @test all(C[nx+NG+1:end, ny+NG+1:end, 1] .== 24.0) # NE
            end

            @testset "RiverBoundary sets concentration and velocity" begin
                # This test should pass
                grid, state = setup_cartesian_test(nx=20, ny=20, nz=1)
                river_indices = 5:10
                state.time = 1.0 
                bcs = [RiverBoundary(side=:West, tracer_name=:C, indices=river_indices, concentration=t->150.0, velocity=t->1.5)]
                apply_boundary_conditions!(state, grid, bcs)
                C = state.tracers[:C]
                for j_phys in river_indices
                    j_glob = j_phys + NG
                    @test all(C[1:NG, j_glob, 1] .== 150.0)
                    @test state.u[NG+1, j_glob, 1] == 1.5
                end
            end
        end
        
        # ... (Other test sets like Physics and SourceSink) ...

    end
end

end # module TestCasesModule

