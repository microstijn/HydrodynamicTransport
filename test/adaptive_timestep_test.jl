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