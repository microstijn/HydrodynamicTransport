# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.HydrodynamicsModule
using ..HydrodynamicTransport.HorizontalTransportModule
using ..HydrodynamicTransport.VerticalTransportModule
using ..HydrodynamicTransport.SourceSinkModule
using ..HydrodynamicTransport.BoundaryConditionsModule
# --- Added for new physics modules ---
using ..HydrodynamicTransport.SettlingModule
using ..HydrodynamicTransport.BedExchangeModule
using ..HydrodynamicTransport.OysterModule
# ------------------------------------
using ProgressMeter
using NCDatasets
using JLD2

# --- Test/Placeholder Versions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; 
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
        loaded_state = JLD2.load_object(restart_from)
        state_to_run = loaded_state
        effective_start_time = loaded_state.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    time_range = effective_start_time:dt:end_time
    
    next_output_time = (output_interval !== nothing) ? state.time + output_interval : Inf
    if output_dir !== nothing; mkpath(output_dir); end
    
    @showprogress "Simulating (Test Mode)..." for (step, time) in enumerate(time_range)
        if time == effective_start_time; continue; end
        state.time = time
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt)
        
        deposition = apply_settling!(state, grid, dt, sediment_params)
        bed_exchange!(state, grid, dt, deposition, sediment_params)

        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        if !isempty(virtual_oysters)
            update_oysters!(state, grid, virtual_oysters, dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
        end

        if output_dir !== nothing && (time >= next_output_time || step == length(time_range))
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            JLD2.save_object(output_filename, state)
            next_output_time += output_interval
        end
    end
    return state
end

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:TVD,
                                  D_crit::Float64=0.0)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    results = [deepcopy(state)]; timesteps = [start_time]
    last_output_time = start_time
    @showprogress "Simulating & Storing (Test Mode)..." for time in time_range
        if time == start_time; continue; end
        state.time = time

        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt)

        deposition = apply_settling!(state, grid, dt, sediment_params)
        bed_exchange!(state, grid, dt, deposition, sediment_params)

        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        if !isempty(virtual_oysters)
            update_oysters!(state, grid, virtual_oysters, dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
        end
        
        if time >= last_output_time + output_interval - 1e-9
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    return results, timesteps
end

# --- Real Data Versions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64; 
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
        loaded_state = JLD2.load_object(restart_from)
        state_to_run = loaded_state
        effective_start_time = loaded_state.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    time_range = effective_start_time:dt:end_time
    
    next_output_time = (output_interval !== nothing) ? state.time + output_interval : Inf
    if output_dir !== nothing; mkpath(output_dir); end
    
    @showprogress "Simulating (Real Data)..." for (step, time) in enumerate(time_range)
        if time == effective_start_time; continue; end
        state.time = time

        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt)
        
        deposition = apply_settling!(state, grid, dt, sediment_params)
        bed_exchange!(state, grid, dt, deposition, sediment_params)

        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        if !isempty(virtual_oysters)
            update_oysters!(state, grid, virtual_oysters, dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
        end

        if output_dir !== nothing && (time >= next_output_time || step == length(time_range))
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            JLD2.save_object(output_filename, state)
            next_output_time += output_interval
        end
    end
    return state
end

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=VirtualOyster[],
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:TVD,
                                  D_crit::Float64=0.0)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    results = [deepcopy(state)]; timesteps = [start_time]
    last_output_time = start_time
    @showprogress "Simulating & Storing (Real Data)..." for time in time_range
        if time == start_time; continue; end
        state.time = time
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt)

        deposition = apply_settling!(state, grid, dt, sediment_params)
        bed_exchange!(state, grid, dt, deposition, sediment_params)

        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        if !isempty(virtual_oysters)
            update_oysters!(state, grid, virtual_oysters, dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
        end

        if time >= last_output_time + output_interval - 1e-9
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    return results, timesteps
end

end # module TimeSteppingModule