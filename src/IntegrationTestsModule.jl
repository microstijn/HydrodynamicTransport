# src/IntegrationTests.jl

module IntegrationTestsModule

export run_integration_tests

using Test
using ..ModelStructs
using ..GridModule
using ..StateModule
using ..TimeSteppingModule
using ..HydrodynamicsModule
using ..VectorOperationsModule # Added for the new test
using NCDatasets

function run_integration_tests()
    @testset "Integration Tests with Real Data (Norway ROMS)" begin
        # Configuration
        netcdf_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
        variable_map = Dict(:u => "u", :v => "v", :time => "ocean_time")
        hydro_data = HydrodynamicData(netcdf_filepath, variable_map)

        @testset "Grid Ingestion and Data Loading" begin
            @info "Running Integration Test: Grid Ingestion..."
            ds = NCDataset(netcdf_filepath)
            
            grid = initialize_curvilinear_grid(hydro_data.filepath)
            state = initialize_state(grid, ds, (:C,))

            update_hydrodynamics!(state, grid, ds, hydro_data, 0.0)
            close(ds)

            @test any(!iszero, state.u)
            @test any(!iszero, state.v)
        end

        @testset "Velocity Rotation with Real Data" begin
            @info "Running Integration Test: Velocity Rotation..."
            ds = NCDataset(netcdf_filepath)
            
            grid = initialize_curvilinear_grid(hydro_data.filepath)
            state = initialize_state(grid, ds, ()) # No tracers needed

            # Load the first time step of velocity data
            update_hydrodynamics!(state, grid, ds, hydro_data, 0.0)
            close(ds)

            # Ensure we have velocities to work with
            @test any(!iszero, state.u)
            @test any(!iszero, state.v)

            # Perform the rotation
            u_east, v_north = rotate_velocities_to_geographic(grid, state.u, state.v)

            # The output should also be populated
            @test any(!iszero, u_east)
            @test any(!iszero, v_north)

            # Check that for a known non-zero angle, the rotation actually changed the vector.
            # We pick a point in the middle of the domain.
            i, j, k = 100, 150, 1
            if grid.angle[i,j] != 0.0
                u_rho_point = 0.5 * (state.u[i, j, k] + state.u[i+1, j, k])
                @test u_east[i, j, k] != u_rho_point
            end
        end

        @testset "Mass Conservation with Point Source" begin
            @info "Running Integration Test: Mass Conservation..."
            ds = NCDataset(netcdf_filepath)
            
            grid_full = initialize_curvilinear_grid(hydro_data.filepath)
            
            # Initialize a 2D state using the surface layer of the dataset dimensions
            grid_2d = CurvilinearGrid(
                grid_full.nx, grid_full.ny, 1,
                grid_full.lon_rho, grid_full.lat_rho, grid_full.lon_u, grid_full.lat_u,
                grid_full.lon_v, grid_full.lat_v, grid_full.z_w[end-1:end],
                grid_full.pm, grid_full.pn, grid_full.angle, grid_full.h,
                grid_full.mask_rho, grid_full.mask_u, grid_full.mask_v,
                grid_full.face_area_x[:,:,end:end], 
                grid_full.face_area_y[:,:,end:end],
                grid_full.volume[:,:,end:end]
            )
            state_2d = initialize_state(grid_2d, (:TestTracer,))

            initial_mass = sum(state_2d.tracers[:TestTracer] .* grid_2d.volume)
            @test initial_mass == 0.0

            source_rate = 1.0e6
            sources = [PointSource(i=100, j=150, k=1, tracer_name=:TestTracer, influx_rate=(t)->source_rate)]
            
            start_time = 0.0
            dt = 120.0
            end_time = 10 * dt

            final_state = run_simulation(grid_2d, state_2d, sources, ds, hydro_data, start_time, end_time, dt)
            close(ds)
            
            final_mass = sum(final_state.tracers[:TestTracer] .* grid_2d.volume)
            
            expected_mass = source_rate * end_time
            
            @test isapprox(final_mass, expected_mass, rtol=0.01)
            # Check that the monotonic scheme isn't creating new mass
            @test final_mass <= expected_mass * (1 + 1e-9) 
        end
    end
end

end # module IntegrationTestsModule