module TimeSteppingModule

export run_simulation

using ..ModelStructs
using ..HydrodynamicsModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ProgressMeter

function run_simulation(grid::Grid, initial_state::State, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time

    @showprogress "Simulating..." for time in time_range
        update_hydrodynamics!(state, grid, hydro_data, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, dt)
    end
    
    return state
end

end # module TimeSteppingModule