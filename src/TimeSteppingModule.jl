# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.HydrodynamicsModule
using ..HydrodynamicTransport.HorizontalTransportModule
using ..HydrodynamicTransport.VerticalTransportModule
using ..HydrodynamicTransport.SourceSinkModule
using ..HydrodynamicTransport.BoundaryConditionsModule
using ProgressMeter
using NCDatasets
using JLD2 # Added for saving state objects

# --- Test/Placeholder Versions ---
"""
    run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; ...)

Runs a simulation using placeholder hydrodynamics.

This function orchestrates the main simulation loop, stepping through time and calling the
necessary physics modules at each step. It is designed for test cases or scenarios where
hydrodynamic data is not read from a file.

# Arguments
- `grid::AbstractGrid`: The computational grid.
- `initial_state::State`: The initial state of the model.
- `sources::Vector{PointSource}`: A vector of point sources for tracers.
- `start_time::Float64`: The simulation start time in seconds.
- `end_time::Float64`: The simulation end time in seconds.
- `dt::Float64`: The time step duration in seconds.

# Keyword Arguments
- `boundary_conditions::Vector{<:BoundaryCondition}`: A vector of boundary conditions to apply.
- `advection_scheme::Symbol`: The advection scheme to use (`:TVD`, `:UP3`, or `:ImplicitADI`). Defaults to `:TVD`.
  `:ImplicitADI` is an unconditionally stable implicit method suitable for large time steps.
- `D_crit::Float64`: The critical water depth for cell-face blocking. If the upstream water
  depth (`grid.h + state.zeta`) is below this value, advective and diffusive fluxes from that
  cell face are blocked. Defaults to `0.0`.
- `output_dir::Union{String, Nothing}`: Directory to save state snapshots. If `nothing`, no output is saved.
- `output_interval::Union{Float64, Nothing}`: Time interval in seconds for saving state snapshots.
- `restart_from::Union{String, Nothing}`: Path to a JLD2 file to restart the simulation from.

# Returns
- `State`: The final state of the model after the simulation.
"""
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; 
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_tracers::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        advection_scheme::Symbol=:TVD,
                        D_crit::Float64=0.0,
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        restart_from::Union{String, Nothing}=nothing)
    
    # --- Logic to handle starting a new simulation vs. restarting ---
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
    
    # --- Setup for file-based output ---
    next_output_time = (output_interval !== nothing) ? state.time + output_interval : Inf
    if output_dir !== nothing
        mkpath(output_dir)
    end
    
    @showprogress "Simulating (Test Mode)..." for (step, time) in enumerate(time_range)
        if time == effective_start_time; continue; end
        state.time = time
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt, sediment_tracers)
        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        # Enforce positivity for all tracers as a safeguard
        for C in values(state.tracers)
            C .= max.(0.0, C)
        end

        # --- Save state to disk if configured ---
        if output_dir !== nothing && (time >= next_output_time || step == length(time_range))
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            JLD2.save_object(output_filename, state)
            next_output_time += output_interval
        end
    end
    return state
end

# This function remains for in-memory storage, useful for small tests
function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_tracers::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
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
        vertical_transport!(state, grid, dt, sediment_tracers)
        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        # Enforce positivity for all tracers as a safeguard
        for C in values(state.tracers)
            C .= max.(0.0, C)
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
"""
    run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64; ...)

Runs a simulation using hydrodynamics from a NetCDF data source.

This function orchestrates the main simulation loop, stepping through time and calling the
necessary physics modules at each step. It is designed for realistic simulations where
hydrodynamic data (like velocity fields and sea surface height) is read from a file.

# Arguments
- `grid::AbstractGrid`: The computational grid.
- `initial_state::State`: The initial state of the model.
- `sources::Vector{PointSource}`: A vector of point sources for tracers.
- `ds::NCDataset`: An opened NetCDF dataset containing the hydrodynamic data.
- `hydro_data::HydrodynamicData`: A struct mapping standard variable names to names in the NetCDF file.
- `start_time::Float64`: The simulation start time in seconds.
- `end_time::Float64`: The simulation end time in seconds.
- `dt::Float64`: The time step duration in seconds.

# Keyword Arguments
- `boundary_conditions::Vector{<:BoundaryCondition}`: A vector of boundary conditions to apply.
- `advection_scheme::Symbol`: The advection scheme to use (`:TVD`, `:UP3`, or `:ImplicitADI`). Defaults to `:TVD`.
  `:ImplicitADI` is an unconditionally stable implicit method suitable for large time steps.
- `D_crit::Float64`: The critical water depth for cell-face blocking. If the upstream water
  depth (`grid.h + state.zeta`) is below this value, advective and diffusive fluxes from that
  cell face are blocked. Defaults to `0.0`.
- `output_dir::Union{String, Nothing}`: Directory to save state snapshots. If `nothing`, no output is saved.
- `output_interval::Union{Float64, Nothing}`: Time interval in seconds for saving state snapshots.
- `restart_from::Union{String, Nothing}`: Path to a JLD2 file to restart the simulation from.

# Returns
- `State`: The final state of the model after the simulation.
"""
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64; 
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_tracers::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        advection_scheme::Symbol=:TVD,
                        D_crit::Float64=0.0,
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        restart_from::Union{String, Nothing}=nothing)
    
    # --- Logic to handle starting a new simulation vs. restarting ---
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
    
    # --- Setup for file-based output ---
    next_output_time = (output_interval !== nothing) ? state.time + output_interval : Inf
    if output_dir !== nothing
        mkpath(output_dir)
    end
    
    @showprogress "Simulating (Real Data)..." for (step, time) in enumerate(time_range)
        if time == effective_start_time; continue; end
        state.time = time

        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt, advection_scheme, D_crit, boundary_conditions)
        vertical_transport!(state, grid, dt, sediment_tracers)
        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        # Enforce positivity for all tracers as a safeguard
        for C in values(state.tracers)
            C .= max.(0.0, C)
        end

        # --- Save state to disk if configured ---
        if output_dir !== nothing && (time >= next_output_time || step == length(time_range))
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            JLD2.save_object(output_filename, state)
            next_output_time += output_interval
        end
    end
    return state
end

# This function remains for in-memory storage
function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; 
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_tracers::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
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
        vertical_transport!(state, grid, dt, sediment_tracers)
        source_sink_terms!(state, grid, sources, functional_interactions, time, dt, D_crit)

        # Enforce positivity for all tracers as a safeguard
        for C in values(state.tracers)
            C .= max.(0.0, C)
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