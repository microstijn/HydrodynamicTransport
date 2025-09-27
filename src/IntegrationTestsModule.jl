# src/IntegrationTests.jl

module IntegrationTestsModule

export run_integration_tests

using Test
using ..ModelStructs
using ..GridModule
using ..StateModule
using ..TimeSteppingModule
using ..HydrodynamicsModule
using NCDatasets

"""
    run_integration_tests()

Runs tests that require the remote Norwegian ROMS dataset to verify the
I/O and simulation pipeline.
"""
function run_integration_tests()
    @testset "Integration Tests with Norwegian ROMS Data" begin
        # --- 1. Configuration ---
        netcdf_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
        
        variable_map = Dict(
            :u => "u", :v => "v", :temp => "temp",
            :salt => "salt", :time => "ocean_time"
        )
        hydro_data = HydrodynamicData(netcdf_filepath, variable_map)

        # --- 2. Test 1: Data Loading Sanity Check ---
        @testset "Data Loading Sanity Check" begin
            ds = NCDataset(netcdf_filepath)
            nx = ds.dim["xi_rho"]; ny = ds.dim["eta_rho"]; nz = ds.dim["s_rho"]
            grid = initialize_grid(nx, ny, nz, Float64(nx*160), Float64(ny*160), 400.0)
            state = initialize_state(grid, (:C,))

            # Call the update function once to load the velocity field for t=0
            HydrodynamicsModule.update_hydrodynamics!(state, grid, ds, hydro_data, 0.0)
            close(ds)

            # Assert that non-zero velocities were actually loaded into the state
            @test any(!iszero, state.u)
            @test any(!iszero, state.v)
        end

        # --- 3. Test 2: Point Source Adds Mass in a Full Run ---
        @testset "Point Source Adds Mass in a Full Run" begin
            ds = NCDataset(netcdf_filepath)
            nx = ds.dim["xi_rho"]; ny = ds.dim["eta_rho"]; nz = 1 # Use 2D for speed

            grid = initialize_grid(nx, ny, nz, Float64(nx*160), Float64(ny*160), 400.0)
            # Start with a tracer field of all zeros
            state = initialize_state(grid, (:TestTracer,))
            initial_mass = sum(state.tracers[:TestTracer] .* grid.volume)
            @test initial_mass == 0.0

            # Configure a constant point source
            source_rate = 100.0 # mass/sec
            sources = [PointSource(i=100, j=150, k=1, tracer_name=:TestTracer, influx_rate=(t)->source_rate)]
            
            # Run a short simulation
            dt = 120.0
            end_time = 10 * dt # Run for 20 minutes

            final_state = run_simulation(grid, state, sources, ds, hydro_data, 0.0, end_time, dt)
            close(ds)
            
            # Validate that the final mass equals the total mass added by the source
            final_mass = sum(final_state.tracers[:TestTracer] .* grid.volume)
            expected_mass = source_rate * end_time
            @test isapprox(final_mass, expected_mass, rtol=1e-9)
        end
    end
end

end # module IntegrationTestsModule