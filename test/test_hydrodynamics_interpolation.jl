# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using HydrodynamicTransport
using NCDatasets

# Import the specific function we want to test
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!

println("--- Test Script for HydrodynamicsModule: Temporal Interpolation ---")

# ==============================================================================
# --- 2. Helper Function to Create a Test NetCDF File ---
# ==============================================================================

function create_test_netcdf(filename::String; nx=1, ny=1, nz=1)
    # Define dimensions
    ds = NCDataset(filename, "c")
    defDim(ds, "xi_u", nx)
    defDim(ds, "eta_u", ny)
    defDim(ds, "s_rho", nz)
    defDim(ds, "ocean_time", 2) # Only two time steps

    # Define time variable
    defVar(ds, "ocean_time", [0.0, 10.0], ("ocean_time",))
    
    # Define u velocity variable (4D: x, y, z, time)
    u_var = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))

    # Write data for the two time steps
    # At t=0, velocity is 1.0
    u_var[:,:,:,1] = fill(1.0, (nx, ny, nz))
    # At t=10, velocity is 3.0
    u_var[:,:,:,2] = fill(3.0, (nx, ny, nz))

    close(ds)
end

# ==============================================================================
# --- 3. Main Test Suite ---
# ==============================================================================

@testset "HydrodynamicsModule: Temporal Interpolation" begin
    
    # Use a temporary directory to ensure the test file is cleaned up
    mktempdir() do temp_dir
        filename = joinpath(temp_dir, "test_hydro.nc")
        create_test_netcdf(filename)

        # --- Setup a minimal test environment ---
        const ng = 2
        grid = CurvilinearGrid(ng, 1, 1, 1, 
            zeros(1+2ng, 1+2ng), zeros(1+2ng, 1+2ng), zeros(1-1+2ng, 1+2ng), zeros(1-1+2ng, 1+2ng), 
            zeros(1+2ng, 1-1+2ng), zeros(1+2ng, 1-1+2ng), [-1.0, 0.0], 
            ones(1+2ng, 1+2ng), ones(1+2ng, 1+2ng), zeros(1+2ng, 1+2ng), ones(1+2ng, 1+2ng),
            trues(1+2ng, 1+2ng), trues(1-1+2ng, 1+2ng), trues(1+2ng, 1-1+2ng),
            zeros(1+1+2ng, 1+2ng, 1), zeros(1+2ng, 1+1+2ng, 1), zeros(1+2ng, 1+2ng, 1))
        
        state = initialize_state(grid, ())
        ds = NCDataset(filename)
        hydro_data = HydrodynamicData(filename, Dict(:u => "u", :time => "ocean_time"))

        # --- Test Cases ---

        @testset "Interpolation at Midpoint" begin
            @info "Testing interpolation at t=5.0..."
            state.time = 5.0
            update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
            
            # Expected value is the average of 1.0 and 3.0, which is 2.0
            # We check the physical interior of the u-array
            u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
            @test all(isapprox.(u_physical, 2.0))
        end

        @testset "Exact Match at First Timestep" begin
            @info "Testing exact match at t=0.0..."
            state.time = 0.0
            update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
            
            # Expected value is exactly 1.0
            u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
            @test all(isapprox.(u_physical, 1.0))
        end

        @testset "Clamping Before Start Time" begin
            @info "Testing clamping at t=-1.0..."
            state.time = -1.0 # Before the first data point
            update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
            
            # Expected behavior is to use the first available data point (t=0, u=1.0)
            u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
            @test all(isapprox.(u_physical, 1.0))
        end

        @testset "Clamping After End Time" begin
            @info "Testing clamping at t=11.0..."
            state.time = 11.0 # After the last data point
            update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
            
            # Expected behavior is to use the last available data point (t=10, u=3.0)
            u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
            @test all(isapprox.(u_physical, 3.0))
        end

        close(ds)
    end
end

println("\n✅ All hydrodynamic interpolation tests passed!")