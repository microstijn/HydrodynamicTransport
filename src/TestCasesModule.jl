# test/runtestsmodule.jl

# --- 1. Set up the Environment ---

using Test
using HydrodynamicTransport
using NCDatasets

# --- Bring all necessary functions into scope for testing ---
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.VectorOperationsModule
using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.HydrodynamicsModule
using HydrodynamicTransport.TimeSteppingModule
using HydrodynamicTransport.UtilsModule


@testset "HydrodynamicTransport.jl Full Test Suite" begin

# ==============================================================================
# --- TESTSET 1: Core Components (Cartesian Grid) ---
# ==============================================================================
@testset "1. Core Components (Cartesian Grid)" begin
    @info "Running Testset 1: Core Components..."
    
    grid = initialize_cartesian_grid(20, 20, 5, 100.0, 100.0, 10.0)
    state = initialize_state(grid, (:C,))
    
    @testset "Initialization" begin
        @test state.tracers[:C] isa Array{Float64, 3}
        @test haskey(state._buffers, :C)
        @test size(state.tracers[:C]) == size(state._buffers[:C])
    end

    @testset "Physics and Mass Conservation" begin
        sources = [PointSource(i=5, j=10, k=1, tracer_name=:C, influx_rate=(t)->100.0)]
        bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West)]
        
        initial_mass = sum(state.tracers[:C] .* grid.volume)
        @test initial_mass == 0.0
        
        dt = 0.5; n_steps = 10
        for _ in 1:n_steps
            state.time += dt
            apply_boundary_conditions!(state, grid, bcs)
            update_hydrodynamics_placeholder!(state, grid, state.time)
            horizontal_transport!(state, grid, dt, :TVD) # Test with default
            vertical_transport!(state, grid, dt)
            source_sink_terms!(state, grid, sources, state.time, dt)
        end
        
        final_mass = sum(state.tracers[:C] .* grid.volume)
        expected_mass_added = 100.0 * dt * n_steps
        @test !any(isnan, state.tracers[:C])
        @test final_mass > 0.0 && final_mass <= expected_mass_added 
    end
end


# ==============================================================================
# --- TESTSET 2: Hydrodynamics Temporal Interpolation ---
# ==============================================================================
@testset "2. Hydrodynamics Temporal Interpolation" begin
    @info "Running Testset 2: Hydrodynamics Temporal Interpolation..."
    
    function create_test_netcdf(filename::String; nx=1, ny=1, nz=1)
        ds = NCDataset(filename, "c")
        defDim(ds, "xi_u", nx); defDim(ds, "eta_u", ny); defDim(ds, "s_rho", nz); defDim(ds, "ocean_time", 2)
        defVar(ds, "ocean_time", [0.0, 10.0], ("ocean_time",))
        u_var = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))
        u_var[:,:,:,1] = fill(1.0, (nx, ny, nz)); u_var[:,:,:,2] = fill(3.0, (nx, ny, nz))
        close(ds)
    end
    
    mktempdir() do temp_dir
        filename = joinpath(temp_dir, "test_hydro.nc")
        create_test_netcdf(filename)

        ng = 2
        grid = CurvilinearGrid(ng, 1, 1, 1, zeros(1+2ng, 1+2ng), zeros(1+2ng, 1+2ng), zeros(1-1+2ng, 1+2ng), zeros(1-1+2ng, 1+2ng), zeros(1+2ng, 1-1+2ng), zeros(1+2ng, 1-1+2ng), [-1.0, 0.0], ones(1+2ng, 1+2ng), ones(1+2ng, 1+2ng), zeros(1+2ng, 1+2ng), ones(1+2ng, 1+2ng), trues(1+2ng, 1+2ng), trues(1-1+2ng, 1+2ng), trues(1+2ng, 1-1+2ng), zeros(1+1+2ng, 1+2ng, 1), zeros(1+2ng, 1+1+2ng, 1), zeros(1+2ng, 1+2ng, 1))
        state = initialize_state(grid, ())
        ds = NCDataset(filename)
        hydro_data = HydrodynamicData(filename, Dict(:u => "u", :time => "ocean_time"))

        # Test midpoint interpolation
        state.time = 5.0
        update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
        u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
        @test all(isapprox.(u_physical, 2.0))

        # Test clamping before start
        state.time = -1.0
        update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
        u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
        @test all(isapprox.(u_physical, 1.0))

        # Test clamping after end
        state.time = 11.0
        update_hydrodynamics!(state, grid, ds, hydro_data, state.time)
        u_physical = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
        @test all(isapprox.(u_physical, 3.0))

        close(ds)
    end
end


# ==============================================================================
# --- TESTSET 3: Curvilinear Grid and Vector Operations ---
# ==============================================================================
@testset "3. Curvilinear Grid and Vector Operations" begin
    @info "Running Testset 3: Vector Rotation Round-Trip..."
    nx, ny, nz, ng = 10, 10, 1, 2
    angle = fill(π/4, (nx + 2*ng, ny + 2*ng))
    dummy_grid = CurvilinearGrid(ng, nx, ny, nz, zeros(nx+2ng, ny+2ng), zeros(nx+2ng, ny+2ng), zeros(nx-1+2ng, ny+2ng), zeros(nx-1+2ng, ny+2ng), zeros(nx+2ng, ny-1+2ng), zeros(nx+2ng, ny-1+2ng), [-1.0, 0.0], ones(nx+2ng, ny+2ng), ones(nx+2ng, ny+2ng), angle, ones(nx+2ng, ny+2ng), trues(nx+2ng, ny+2ng), trues(nx-1+2ng, ny+2ng), trues(nx+2ng, ny-1+2ng), zeros(nx+1+2ng, ny+2ng, nz), zeros(nx+2ng, ny+1+2ng, nz), zeros(nx+2ng, ny+2ng, nz))
    state = initialize_state(dummy_grid, ())

    u_east_initial = ones(Float64, nx, ny, nz)
    v_north_initial = zeros(Float64, nx, ny, nz)

    rotate_velocities_to_grid!(state.u, state.v, dummy_grid, u_east_initial, v_north_initial)
    u_east_final, v_north_final = rotate_velocities_to_geographic(dummy_grid, state.u, state.v)

    @test u_east_final ≈ u_east_initial
    @test v_north_final ≈ v_north_initial atol=1e-12
end


# ==============================================================================
# --- TESTSET 4 & 5: File I/O, Utilities, and Full Simulations ---
# ==============================================================================
# We use a temporary directory to ensure our test is clean and leaves no files behind
mktempdir() do temp_dir
    @info "Running Testsets 4 & 5 in temporary directory: $temp_dir"
    
    # --- Setup: Generate the test dataset ---
    braided_river_file = joinpath(temp_dir, "braided_river_data.nc")
    
    # We bring the dataset generator function into this scope
    # In a real package, this might be in a separate test/test_utils.jl file
    include(joinpath(@__DIR__, "run_braided_river_test.jl"))
    generate_and_write_dataset(braided_river_file, "temp_plot.png")

    @testset "4. File I/O and Utility Functions" begin
        @info "Running Testset 4: Utilities on generated NetCDF..."
        @test isfile(braided_river_file)
        hydro_data = create_hydrodynamic_data_from_file(braided_river_file)
        @test hydro_data isa HydrodynamicData && hydro_data.var_map[:u] == "u"
        safe_dt = estimate_stable_timestep(braided_river_file)
        @test safe_dt > 0 && safe_dt < 30.0
    end

    @testset "5. End-to-End Simulations" begin
        ds = NCDataset(braided_river_file)
        grid = initialize_curvilinear_grid(braided_river_file)
        hydro_data = create_hydrodynamic_data_from_file(braided_river_file)
        sources = [PointSource(i=45, j=40, k=1, tracer_name=:Tracer, influx_rate=(t)->1.0e4)]
        bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West)]
        dt = estimate_stable_timestep(braided_river_file)
        
        @testset "TVD Scheme (Default)" begin
            @info "Running End-to-End Test with :TVD scheme..."
            state_tvd = initialize_state(grid, ds, (:Tracer,))
            final_state_tvd = run_simulation(grid, state_tvd, sources, ds, hydro_data, 0.0, 3*3600.0, dt, boundary_conditions=bcs, advection_scheme=:TVD)
            
            @test !any(isnan, final_state_tvd.tracers[:Tracer])
            mask = grid.mask_rho
            land_indices = findall(iszero, mask)
            @test all(final_state_tvd.tracers[:Tracer][land_indices] .== 0.0)
        end

        @testset "UP3 Scheme" begin
            @info "Running End-to-End Test with :UP3 scheme..."
            state_up3 = initialize_state(grid, ds, (:Tracer,))
            final_state_up3 = run_simulation(grid, state_up3, sources, ds, hydro_data, 0.0, 3*3600.0, dt, boundary_conditions=bcs, advection_scheme=:UP3)
            
            @test !any(isnan, final_state_up3.tracers[:Tracer])
            mask = grid.mask_rho
            land_indices = findall(iszero, mask)
            @test all(final_state_up3.tracers[:Tracer][land_indices] .== 0.0)
        end

        close(ds)
    end
end

println("\n✅ ✅ ✅ HydrodynamicTransport.jl: All tests passed successfully! ✅ ✅ ✅")

end # End of the full test suite