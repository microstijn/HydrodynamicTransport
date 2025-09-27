# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..ModelStructs
using ..HydrodynamicsModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ProgressMeter
using NCDatasets

# --------------------------------------------------------------------------
# --- Test/Placeholder Versions ---
# --------------------------------------------------------------------------

"""
    run_simulation(grid, initial_state, sources, start_time, end_time, dt)
"""
function run_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time

    @showprogress "Simulating (Test Mode)..." for time in time_range
        # --- FIX: Skip the update step at t=0, which is the initial state ---
        if time == start_time
            continue
        end
        
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    
    return state
end

"""
    run_and_store_simulation(grid, initial_state, sources, start_time, end_time, dt, output_interval)
"""
function run_and_store_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    
    results = [deepcopy(state)]; timesteps = [start_time]; last_output_time = start_time
    
    @showprogress "Simulating (Test Mode)..." for time in time_range
        if time == start_time; continue; end
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)

        if time >= last_output_time + output_interval
            push!(results, deepcopy(state)); push!(timesteps, time); last_output_time = time
        end
    end
    
    return results, timesteps
end


# --------------------------------------------------------------------------
# --- Real Data Versions ---
# --------------------------------------------------------------------------

"""
    run_simulation(grid, initial_state, sources, ds, hydro_data, start_time, end_time, dt)
"""
function run_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time

    @showprogress "Simulating (Real Data)..." for time in time_range
        # --- FIX: Skip the update step at t=0, which is the initial state ---
        if time == start_time
            continue
        end

        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    
    return state
end


"""
    run_and_store_simulation(grid, initial_state, sources, ds, hydro_data, start_time, end_time, dt, output_interval)
"""
function run_and_store_simulation(grid::Grid, initial_state::State, sources::Vector{PointSource}, ds::NCDataset, hydro_data::HydrodynamicData, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64)
    state = deepcopy(initial_state)
    time_range = start_time:dt:end_time
    
    results = [deepcopy(state)]
    timesteps = [start_time]
    last_output_time = start_time
    
    @showprogress "Simulating (Real Data)..." for time in time_range
        if time == start_time; continue; end

        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        horizontal_transport!(state, grid, dt)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)

        if time >= last_output_time + output_interval
            push!(results, deepcopy(state))
            push!(timesteps, time)
            last_output_time = time
        end
    end
    
    return results, timesteps
end

end # module TimeSteppingModule