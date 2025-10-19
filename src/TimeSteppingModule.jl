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
using ProgressMeter
using NCDatasets
using JLD2

# --- Test/Placeholder Versions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64;
                        use_adaptive_dt::Bool=false,
                        cfl_max::Float64=0.8,
                        dt_max::Float64=dt,
                        dt_min::Float64=0.1,
                        dt_growth_factor::Float64=1.1,
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                        oyster_tracers::NamedTuple=NamedTuple(),
                        advection_scheme::Symbol=:TVD,
                        D_crit::Float64=0.0,
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        restart_from::Union{String, Nothing}=nothing)
    
    local state_to_run, effective_start_time
    if restart_from !== nothing
        println("--- Restarting simulation from checkpoint: $restart_from ---")
        state_to_run = JLD2.load(restart_from, "state")
        effective_start_time = state_to_run.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    time = effective_start_time
    current_dt = dt

    next_output_time = if output_interval !== nothing
        ceil((time + 1e-9) / output_interval) * output_interval
    else
        Inf
    end
    if output_dir !== nothing; mkpath(output_dir); end
    
    pbar = Progress(floor(Int, end_time - time); desc="Simulating (Test Mode)...", dt=1.0)

    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min; println("\nWarning: Timestep below minimum."); break; end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
            horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
            vertical_transport!(state_backup, grid, trial_dt)
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(virtual_oysters)
                update_oysters!(state_backup, grid, virtual_oysters, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end

            cfl_actual = use_adaptive_dt ? calculate_max_cfl_term(state_backup, grid) * trial_dt : 0.0
            
            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / cfl_actual)
            else
                state = state_backup
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
        
        if output_dir !== nothing && abs(time - next_output_time) < 1e-9
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            jldsave(output_filename; state=state, virtual_oysters=virtual_oysters)
            next_output_time += output_interval
        end

        ProgressMeter.update!(pbar, floor(Int, time - effective_start_time))
    end
    ProgressMeter.finish!(pbar)
    return state
end

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  use_adaptive_dt::Bool=false, cfl_max::Float64=0.8, dt_max::Float64=dt, dt_min::Float64=0.1, dt_growth_factor::Float64=1.1,
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:TVD,
                                  D_crit::Float64=0.0)
    state = deepcopy(initial_state)
    time = start_time
    current_dt = dt
    results = [(state=deepcopy(state), oysters=deepcopy(virtual_oysters))]; timesteps = [start_time]
    next_output_time = start_time + output_interval

    pbar = Progress(floor(Int, end_time - time); desc="Simulating & Storing (Test Mode)...", dt=1.0)

    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min; println("\nWarning: Timestep below minimum."); break; end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
            horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
            vertical_transport!(state_backup, grid, trial_dt)
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            cfl_actual = use_adaptive_dt ? calculate_max_cfl_term(state_backup, grid) * trial_dt : 0.0
            
            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / cfl_actual)
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

# --- Real Data Versions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64; 
                        use_adaptive_dt::Bool=false, cfl_max::Float64=0.8, dt_max::Float64=dt, dt_min::Float64=0.1, dt_growth_factor::Float64=1.1,
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                        oyster_tracers::NamedTuple=NamedTuple(),
                        advection_scheme::Symbol=:TVD,
                        D_crit::Float64=0.0,
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        restart_from::Union{String, Nothing}=nothing)
    
    local state_to_run, effective_start_time
    if restart_from !== nothing
        println("--- Restarting simulation from checkpoint: $restart_from ---")
        restart_data = JLD2.load(restart_from)
        state_to_run = restart_data["state"]
        # If oysters are in the restart file, load them. Otherwise, use the initial ones.
        virtual_oysters = get(restart_data, "virtual_oysters", virtual_oysters)
        effective_start_time = state_to_run.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    time = effective_start_time
    current_dt = dt

    #  Initialize tracking variables ---
    min_dt_taken = Inf
    max_dt_taken = 0.0
    start_wall_time = time_ns() # Record the real-world start time

    next_output_time = if output_interval !== nothing
        ceil((time + 1e-9) / output_interval) * output_interval
    else
        Inf
    end
    if output_dir !== nothing; mkpath(output_dir); end
    
    pbar = Progress(floor(Int, end_time - time); desc="Simulating (Real Data)...", dt=1.0)

    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min; println("\nWarning: Timestep below minimum."); break; end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            update_hydrodynamics!(state_backup, grid, ds, hydro_data, time + trial_dt)
            horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
            vertical_transport!(state_backup, grid, trial_dt)
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            cfl_actual = use_adaptive_dt ? calculate_max_cfl_term(state_backup, grid) * trial_dt : 0.0

            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / cfl_actual)
            else
                state = state_backup
                virtual_oysters = oysters_backup
                step_successful = true

                # Update min/max dt trackers
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

        # Pass all new values to the progress bar
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

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  use_adaptive_dt::Bool=false, cfl_max::Float64=0.8, dt_max::Float64=dt, dt_min::Float64=0.1, dt_growth_factor::Float64=1.1,
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:TVD,
                                  D_crit::Float64=0.0)
    state = deepcopy(initial_state)
    time = start_time
    current_dt = dt
    results = [(state=deepcopy(state), oysters=deepcopy(virtual_oysters))]; timesteps = [start_time]
    next_output_time = start_time + output_interval

    pbar = Progress(floor(Int, end_time - time); desc="Simulating & Storing (Real Data)...", dt=1.0)
    
    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
        trial_dt = min(trial_dt, end_time - time, next_output_time - time)

        if use_adaptive_dt && trial_dt < dt_min; println("\nWarning: Timestep below minimum."); break; end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            update_hydrodynamics!(state_backup, grid, ds, hydro_data, time + trial_dt)
            horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions)
            vertical_transport!(state_backup, grid, trial_dt)
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            cfl_actual = use_adaptive_dt ? calculate_max_cfl_term(state_backup, grid) * trial_dt : 0.0

            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / cfl_actual)
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
        ProgressMeter.update!(pbar, floor(Int, time - effective_start_time); showvalues = [
            (:wall_time, round(time / 3600, digits=1)),
            (:adaptive_timestep, round(trial_dt, digits=2))
        ])
    end
    ProgressMeter.finish!(pbar)
    return results, timesteps
end

end # module TimeSteppingModule