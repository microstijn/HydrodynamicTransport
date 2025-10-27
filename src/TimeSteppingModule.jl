# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.HydrodynamicsModule
using ..HydrodynamicTransport.HorizontalTransportModule
using ..HydrodynamicTransport.VerticalTransportModule
using ..HydrodynamicTransport.SourceSinkModule
using ..HydrodynamicTransport.BoundaryConditionsModule
using ..HydrodynamicTransport.SettlingModule
using ..HydrodynamicTransport.BedExchangeModule
using ..HydrodynamicTransport.OysterModule
using ..HydrodynamicTransport.UtilsModule: calculate_max_cfl_term
using ..HydrodynamicTransport.FluxLimitersModule 
using ProgressMeter
using NCDatasets
using JLD2

# --- Import the NEW TVD functions ---
using ..HorizontalTransportModule: advect_diffuse_tvd_implicit_x!, advect_diffuse_tvd_implicit_y!
using ..VerticalTransportModule: advect_diffuse_tvd_implicit_z!

"""
    run_simulation(grid, initial_state, sources, start_time, end_time, dt; ...)

Main driver for running a hydrodynamic transport simulation.
... (docstring arguments) ...
- `limiter_func::Function`: The flux limiter function to use (e._g., `van_leer`, `minmod`).
"""
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64;
                        ds::Union{NCDataset, Nothing}=nothing,
                        hydro_data::Union{HydrodynamicData, Nothing}=nothing,
                        use_adaptive_dt::Bool=false,
                        cfl_max::Float64=0.8,
                        dt_max::Float64=dt,
                        dt_min::Float64=0.1,
                        dt_growth_factor::Float64=1.1,
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        virtual_oysters::Vector{VirtualOyster}=Vector{VirtualOyster}(),
                        oyster_tracers::NamedTuple=NamedTuple(),
                        advection_scheme::Symbol=:TVD,
                        limiter_func::Function=FluxLimitersModule.van_leer, # <-- NEW ARGUMENT
                        Kh::Float64=1.0,
                        Kz::Float64=1e-4,
                        D_crit::Float64=0.0,
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        restart_from::Union{String, Nothing}=nothing)

    local state_to_run, effective_start_time
    if restart_from !== nothing
        @info "Restarting simulation from checkpoint: $restart_from"
        restart_data = JLD2.load(restart_from)
        state_to_run = restart_data["state"]
        virtual_oysters = get(restart_data, "virtual_oysters", virtual_oysters)
        effective_start_time = state_to_run.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    time = effective_start_time
    current_dt = dt

    min_dt_taken = Inf
    max_dt_taken = 0.0
    start_wall_time = time_ns()

    next_output_time = if output_interval !== nothing
        ceil((time + 1e-9) / output_interval) * output_interval
    else
        Inf
    end
    if output_dir !== nothing; mkpath(output_dir); end
    
    desc_str = (ds !== nothing) ? "Simulating..." : "Simulating (test mode)..."
    pbar = Progress(floor(Int, end_time - time); desc=desc_str, dt=1.0)

    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min
            @warn "\nWarning: Timestep below minimum threshold. Stopping simulation."
            break
        end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            if  time == start_time
                apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            end
            
            # Hydrodynamics Step 
            if ds !== nothing && hydro_data !== nothing
                update_hydrodynamics!(state_backup, grid, ds, hydro_data, time + trial_dt)
            else
                update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
            end

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)

            # Transport Step 
            if advection_scheme == :ImplicitADI_3D
                for tracer_name in keys(state_backup.tracers)
                    C_initial = state_backup.tracers[tracer_name]
                    C_buffer1 = state_backup._buffer1[tracer_name]
                    C_buffer2 = state_backup._buffer2[tracer_name]

                    advect_diffuse_tvd_implicit_x!(C_buffer1, C_initial, state_backup, grid, trial_dt, Kh, limiter_func, D_crit)
                    advect_diffuse_tvd_implicit_y!(C_buffer2, C_buffer1, state_backup, grid, trial_dt, Kh, limiter_func, D_crit)
                    advect_diffuse_tvd_implicit_z!(C_initial, C_buffer2, state_backup, grid, trial_dt, Kz, limiter_func, D_crit)
                end
            else
                horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
                vertical_transport!(state_backup, grid, trial_dt)
            end

            # --- APPLY CLIPPING IMMEDIATELY AFTER TRANSPORT ---
            #=
            for C_array in values(state_backup.tracers)
                clamp!(C_array, 0.0, Inf)
            end
            =#

            # --- Physics Steps ---
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            # --- Timestep Validation ---
            cfl_actual = calculate_max_cfl_term(state_backup, grid) * trial_dt
            
            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / (cfl_actual + 1e-9))
            else
                state = state_backup
                virtual_oysters = oysters_backup
                step_successful = true

                min_dt_taken = min(min_dt_taken, trial_dt)
                max_dt_taken = max(max_dt_taken, trial_dt)

                if use_adaptive_dt && cfl_actual < 0.5 * cfl_max
                    current_dt = min(dt_max, trial_dt * dt_growth_factor)
                else
                    current_dt = trial_dt
                end
            end
        end
        
        time += trial_dt
        state.time = time
        
        if output_dir !== nothing && abs(time - next_output_time) < 1e-9
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            jldsave(output_filename; state=state, virtual_oysters=virtual_oysters)
            next_output_time += output_interval
        end

        elapsed_wall_time_min = (time_ns() - start_wall_time) / 1e9 / 60
        ProgressMeter.update!(pbar, floor(Int, time - effective_start_time); showvalues = [
            (:sim_time_h, round(time / 3600, digits=1)),
            (:wall_time_m, round(elapsed_wall_time_min, digits=1)),
            (:current_timestep_s, round(trial_dt, digits=2)),
            (:min_timestep_s, round(min_dt_taken, digits=2)),
            (:max_timestep_s, round(max_dt_taken, digits=2))
        ])
    end
    ProgressMeter.finish!(pbar)
    return state
end


"""
    run_and_store_simulation(grid, initial_state, sources, start_time, end_time, dt, output_interval; ...)

Runs a simulation and stores the state at specified intervals in memory.
...
"""
function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64;
                                  ds::Union{NCDataset, Nothing}=nothing,
                                  hydro_data::Union{HydrodynamicData, Nothing}=nothing,
                                  use_adaptive_dt::Bool=false,
                                  cfl_max::Float64=0.8,
                                  dt_max::Float64=dt,
                                  dt_min::Float64=0.1,
                                  dt_growth_factor::Float64=1.1,
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=Vector{VirtualOyster}(),
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:TVD,
                                  limiter_func::Function=FluxLimitersModule.van_leer, # <-- NEW ARGUMENT
                                  Kh::Float64=1.0,
                                  Kz::Float64=1e-4,
                                  D_crit::Float64=0.0)
                                  
    state = deepcopy(initial_state)
    time = start_time
    current_dt = dt
    results = [(state=deepcopy(state), oysters=deepcopy(virtual_oysters))]
    timesteps = [start_time]
    next_output_time = start_time + output_interval

    desc_str = (ds !== nothing) ? "Simulating & Storing (Real Data)..." : "Simulating & Storing (Test Mode)..."
    pbar = Progress(floor(Int, end_time - time); desc=desc_str, dt=1.0)
    
    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min
            @warn "\nWarning: Timestep below minimum threshold. Stopping simulation."
            break
        end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            if time == start_time
                apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            end

            if ds !== nothing && hydro_data !== nothing
                update_hydrodynamics!(state_backup, grid, ds, hydro_data, time + trial_dt)
            else
                # updates it with whatever is currently encoded in the placeholder 
                # probably a vortex
                update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
            end

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)

            if advection_scheme == :ImplicitADI_3D
                # if we do advection/diffusion in one go we no longer need to call for them seperately
                for tracer_name in keys(state_backup.tracers)
                    C_initial = state_backup.tracers[tracer_name]
                    C_buffer1 = state_backup._buffer1[tracer_name]
                    C_buffer2 = state_backup._buffer2[tracer_name]
                    
                    # Call the new TVD functions
                    advect_diffuse_tvd_implicit_x!(C_buffer1, C_initial, state_backup, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_y!(C_buffer2, C_buffer1, state_backup, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_z!(C_initial, C_buffer2, state_backup, grid, trial_dt, Kz, limiter_func)
                end
            else
                horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
                vertical_transport!(state_backup, grid, trial_dt)
            end

            # Clipping logic. Small negative values can snowball. 
            for C_array in values(state_backup.tracers)
                clamp!(C_array, 0.0, Inf)
            end
            
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            cfl_actual = calculate_max_cfl_term(state_backup, grid) * trial_dt
            
            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / (cfl_actual + 1e-9))
            else
                state = state_backup
                virtual_oysters = oysters_backup
                step_successful = true
                if use_adaptive_dt && cfl_actual < 0.5 * cfl_max
                    current_dt = min(dt_max, trial_dt * dt_growth_factor)
                else
                    current_dt = trial_dt
                end
            end
        end
        
        time += trial_dt
        state.time = time
        
        if abs(time - next_output_time) < 1e-9
            push!(results, (state=deepcopy(state), oysters=deepcopy(virtual_oysters)))
            push!(timesteps, time)
            next_output_time += output_interval
        end
        ProgressMeter.update!(pbar, floor(Int, time - start_time))
    end
    ProgressMeter.finish!(pbar)
    return results, timesteps
end

end # module TimeSteppingModule