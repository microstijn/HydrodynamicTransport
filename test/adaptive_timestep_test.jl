# test/adaptive_timestep_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using Test
using UnicodePlots

# --- Explicitly import unexported functions needed for the custom loop ---
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.SourceSinkModule: source_sink_terms!
using HydrodynamicTransport.UtilsModule: calculate_max_cfl_term

@testset "Adaptive Timestep" begin
    println("Running integration test for adaptive timestepping...")

    # --- 1. Setup a Grid and Time-Varying Flow ---
    ng = 2; nx, ny, nz = 20, 20, 1
    dx, dy = 10.0, 10.0
    grid = initialize_cartesian_grid(nx, ny, nz, nx*dx, ny*dy, nz*10.0; ng=ng)
    
    function generate_time_varying_flow!(state::State, time::Float64, total_time::Float64)
        time_fraction = time / total_time
        magnitude = 4.0 * time_fraction * (1.0 - time_fraction) # Peaks at 1.0 when time_fraction is 0.5
        max_velocity = 2.5 # m/s
        
        for k in 1:nz, j in 1:ny, i in 1:(nx + 1)
            state.u[i+ng, j+ng, k] = max_velocity * magnitude * ((j-0.5)/ny)
        end
        state.v .= 0.0
    end

    # --- 2. Simulation Parameters ---
    sources = [PointSource(i=5, j=10, k=1, tracer_name=:TestTracer, influx_rate=(t)->100.0)]
    bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West)]
    start_time = 0.0
    end_time = 100.0

    @testset "2. Adaptive timestep (Expects Success)" begin
        println("  -> Running with adaptive timestepping...")
        state = initialize_state(grid, (:TestTracer,))
        
        dt_init = 5.0
        cfl_max = 0.8
        dt_max = dt_init
        dt_min = 0.1
        dt_growth_factor = 1.2
        
        time = start_time
        current_dt = dt_init
        dt_history = Float64[]

        while time < end_time
            trial_dt = min(current_dt, dt_max, end_time - time)
            if trial_dt < dt_min; break; end

            step_successful = false
            while !step_successful
                state_backup = deepcopy(state)
                
                generate_time_varying_flow!(state_backup, time + trial_dt, end_time)
                horizontal_transport!(state_backup, grid, trial_dt, :TVD, 0.0, bcs)
                
                cfl_actual = calculate_max_cfl_term(state_backup, grid) * trial_dt

                if cfl_actual > cfl_max
                    trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / cfl_actual)
                else
                    state = state_backup
                    step_successful = true
                    if cfl_actual < 0.3 * cfl_max
                        current_dt = min(dt_max, trial_dt * dt_growth_factor)
                    else
                        current_dt = trial_dt
                    end
                end
            end
            
            time += trial_dt
            state.time = time
            push!(dt_history, trial_dt)
        end

        @test !any(isnan, state.tracers[:TestTracer])
        @test minimum(dt_history) < dt_init
        @test dt_history[end] > minimum(dt_history)

        println("\n--- Adaptive Timestep Evolution ---")
        println(lineplot(dt_history, title="dt (s) vs. Timestep Number", width=70, ylim=(0, dt_max+1)))
        println("Initial dt: $dt_init, Min dt reached: $(minimum(dt_history)), Final dt: $(dt_history[end])")
    end
end



# ==============================================================================
# --- TESTSET 9: CFL Calculation ---
# ==============================================================================
@testset "9. CFL Calculation" begin
    @info "Running Testset 9: CFL Calculation..."

    @testset "CartesianGrid 3D CFL" begin
        # --- 1. Setup ---
        nx, ny, nz = 10, 10, 10
        Lx, Ly, Lz = 100.0, 100.0, 100.0
        grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
        state = initialize_state(grid, (:C,))
        ng = grid.ng

        dx = Lx / nx # 10.0
        dz = Lz / nz # 10.0
        
        # --- 2. Velocities ---
        i, j = 5, 5
        i_glob, j_glob = i + ng, j + ng

        # Set velocities to 0
        state.u .= 0.0
        state.v .= 0.0
        state.w .= 0.0

        # A) Fast horizontal velocity at the BOTTOM (k=1)
        u_fast_val = 40.0
        state.u[i_glob, j_glob, 1] = u_fast_val
        # u_center = 0.5 * (40.0 + 0.0) = 20.0
        # cfl_horiz = 20.0 / dx = 2.0

        # B) Fast vertical velocity at the face k=2
        w_fast_val = 30.0
        state.w[i_glob, j_glob, 2] = w_fast_val
        # w_face_vel = max(abs(w[k=1]), abs(w[k=2])) = max(0.0, 30.0) = 30.0
        # cfl_vert = 30.0 / dz = 3.0
        
        # --- 3. Calculate Expected Result ---
        # The new function at k=1 should find:
        # total_cfl = cfl_horiz (2.0) + cfl_vert (3.0) = 5.0
        # All other cells will be smaller.
        expected_max_cfl = 5.0
        
        # --- 4. Run and Test ---
        result = calculate_max_cfl_term(state, grid)
        @test result == expected_max_cfl
    end

    @testset "CurvilinearGrid 3D CFL" begin
        # --- 1. Setup ---
        nx, ny, nz = 10, 10, 10
        Lx, Ly, Lz = 100.0, 100.0, 100.0
        dx = Lx / nx; dy = Ly / ny
        ng = 2
        
        nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
        pm_full = fill(1.0 / dx, (nx_tot, ny_tot)) # pm = 0.1
        pn_full = fill(1.0 / dy, (nx_tot, ny_tot)) # pn = 0.1
        z_w_vec = collect(range(-Lz, 0.0, length=nz+1))
        dz = abs(z_w_vec[2] - z_w_vec[1]) # dz = 10.0

        face_area_x = fill(dy * dz, (nx_tot + 1, ny_tot, nz))
        face_area_y = fill(dx * dz, (nx_tot, ny_tot + 1, nz))
        volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))

        curv_grid = CurvilinearGrid(
            ng, nx, ny, nz,
            zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot), zeros(nx_tot+1, ny_tot), zeros(nx_tot+1, ny_tot), zeros(nx_tot, ny_tot+1), zeros(nx_tot, ny_tot+1),
            z_w_vec, pm_full, pn_full, zeros(nx_tot, ny_tot), fill(Lz, (nx_tot, ny_tot)),
            ones(Bool, nx_tot, ny_tot), ones(Bool, nx_tot+1, ny_tot), ones(Bool, nx_tot, ny_tot+1),
            face_area_x, face_area_y, volume
        )
        state = initialize_state(curv_grid, (:C,))
        
        # --- 2. Velocities (same as Cartesian) ---
        i, j = 5, 5
        i_glob, j_glob = i + ng, j + ng

        # Set velocities to 0
        state.u .= 0.0
        state.v .= 0.0
        state.w .= 0.0

        # A) Fast horizontal velocity at the BOTTOM (k=1)
        u_fast_val = 40.0
        state.u[i_glob, j_glob, 1] = u_fast_val
        # u_center = 0.5 * (40.0 + 0.0) = 20.0
        # cfl_horiz = 20.0 * pm = 2.0

        # B) Fast vertical velocity at the face k=2
        w_fast_val = 30.0
        state.w[i_glob, j_glob, 2] = w_fast_val
        # w_face_vel = max(abs(w[k=1]), abs(w[k=2])) = max(0.0, 30.0) = 30.0
        # cfl_vert = 30.0 / dz = 3.0
        
        # --- 3. Calculate Expected Result ---
        expected_max_cfl = 5.0
        
        # --- 4. Run and Test ---
        result = calculate_max_cfl_term(state, curv_grid)
        @test result == expected_max_cfl
    end
    
end