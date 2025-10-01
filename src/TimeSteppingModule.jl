# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..ModelStructs
using ..HydrodynamicsModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ..BoundaryConditionsModule # Import the new module
using ProgressMeter
using NCDatasets

function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}())
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    results = [deepcopy(state)]; timesteps = [start_time]
    last_output_time = start_time
    
    @showprogress "Simulating & Storing..." for time in time_range
        if time == start_time; continue; end
        
        # Update the state's internal time
        state = State(state.tracers, state.u, state.v, state.w, state.temperature, state.salinity, state.tss, state.uvb, time)
        
        # --- NEW: Apply boundary conditions by filling ghost cells ---
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

# The run_simulation function would be updated similarly
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64; boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}())
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    
    @showprogress "Simulating..." for time in time_range
        if time == start_time; continue; end
        state = State(state.tracers, state.u, state.v, state.w, state.temperature, state.salinity, state.tss, state.uvb, time)
        
        apply_boundary_conditions!(state, grid, boundary_conditions)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    return state
end


end # module TimeSteppingModule