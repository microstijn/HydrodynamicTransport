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

# --- Test/Placeholder Versions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}())
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    @showprogress "Simulating (Test Mode)..." for time in time_range
        if time == start_time; continue; end
        state.time = time
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    return state
end

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}())
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    results = [deepcopy(state)]; timesteps = [start_time]
    last_output_time = start_time
    @showprogress "Simulating & Storing (Test Mode)..." for time in time_range
        if time == start_time; continue; end
        state.time = time

        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
        
        if time >= last_output_time + output_interval - 1e-9
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    return results, timesteps
end

# --- NEW: Real Data Versions with Boundary Conditions ---
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(), ds::NCDataset, hydro_data::HydrodynamicData)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    @showprogress "Simulating (Real Data)..." for time in time_range
        if time == start_time; continue; end
        state.time = time

        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    return state
end

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(), ds::NCDataset, hydro_data::HydrodynamicData)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    results = [deepcopy(state)]; timesteps = [start_time]
    last_output_time = start_time
    @showprogress "Simulating & Storing (Real Data)..." for time in time_range
        if time == start_time; continue; end
        state.time = time
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)

        if time >= last_output_time + output_interval - 1e-9
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    return results, timesteps
end

end # module TimeSteppingModule