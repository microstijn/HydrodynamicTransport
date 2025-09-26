# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..ModelStructs
using ..HydrodynamicsModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ProgressMeter

"""
    run_simulation(grid, initial_state, sources, start_time, end_time, dt)

The main driver for the simulation, now with a flexible source configuration.
Returns only the final state of the simulation.
"""
function run_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time

    @showprogress "Simulating..." for time in time_range
        update_hydrodynamics!(state, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        # --- UPDATED CALL ---
        # Pass the sources vector and current time to the function
        source_sink_terms!(state, grid, sources, time, dt)
    end
    
    return state
end


"""
    run_and_store_simulation(grid, initial_state, sources, start_time, end_time, dt, output_interval)

Runs the simulation with a flexible source configuration and stores the state at regular intervals.
"""
function run_and_store_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    
    # Prepare storage for the results
    results = [deepcopy(state)]
    timesteps = [start_time]
    
    last_output_time = start_time
    
    @showprogress "Simulating and Storing..." for time in time_range
        if time == start_time
            continue # Skip the first step as it's already stored
        end

        update_hydrodynamics!(state, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        # --- UPDATED CALL ---
        # Pass the sources vector and current time to the function
        source_sink_terms!(state, grid, sources, time, dt)

        # Check if it's time to save the output
        if time >= last_output_time + output_interval
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    
    return results, timesteps
end

end # module TimeSteppingModule