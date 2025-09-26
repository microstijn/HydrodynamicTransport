# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation

using ..ModelStructs
using ..HydrodynamicsModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ProgressMeter

"""
    run_simulation(grid, state, start_time, end_time, dt)

The main driver for the simulation. This version is for self-contained runs
and uses the placeholder hydrodynamics.
"""
function run_simulation(grid::Grid, initial_state::State, start_time::Float64, end_time::Float64, dt::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time

    @showprogress "Simulating..." for time in time_range
        # Calls the simplified placeholder hydrodynamics function
        update_hydrodynamics!(state, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, dt)
    end
    
    return state
end

end # module TimeSteppingModule