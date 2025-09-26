# src/TimeStepping.jl

module TimeSteppingModule

export run_simulation

using ..ModelStructs
using ..HorizontalTransportModule
using Logging  
using Dates

# helper

"""
    format_elapsed_time(start_time::Float64) -> String

Calculates the time elapsed since `start_time` and returns it as a
formatted "HH:MM:SS" string.
"""
function format_elapsed_time(start_time::Float64)
    elapsed_seconds = time() - start_time
    time_obj = Time(0) + Dates.Second(round(Int, elapsed_seconds))
    return Dates.format(time_obj, "HH:MM:SS")
end


# STUBBY STUBS

function update_hydrodynamics!(state::State, grid::Grid, time::Float64)
    # TODO: Implement reading of velocity, temp, etc., from files for the given time.
    return nothing
end

function vertical_transport!(state::State, grid::Grid, dt::Float64)
    # TODO: Implement vertical advection and diffusion here.
    return nothing
end

function source_sink_terms!(state::State, grid::Grid, dt::Float64)
    # TODO: This is where the previrS reaction module will be called for each cell.
    return nothing
end


"""
    run_simulation(grid::Grid, state::State, start_time::Float64, end_time::Float64, dt::Float64)

The main driver for the hydrodynamic transport simulation.

This function orchestrates the time-stepping loop, applying the operator-splitting
methodology to update the model state over a defined simulation period.

# Arguments
- `grid::Grid`: The static grid object for the simulation.
- `state::State`: The initial state of the simulation.
- `start_time::Float64`: The starting time of the simulation (e.g., in hours).
- `end_time::Float64`: The ending time of the simulation (in hours).
- `dt::Float64`: The time step for each iteration (in hours).
"""
function run_simulation(grid::Grid, state::State, start_time::Float64, end_time::Float64, dt::Float64)
    
    start_wall_time = time() 

    @info "Starting Simulation" 
    
    for time_sim  in start_time:dt:end_time

        wall_time_str = format_elapsed_time(start_wall_time)

        @info "Simulating time: $(round(time_sim, digits=2)) hours (Wall time: $(wall_time_str))"

        # Operator Splitting Steps

        # 1. Update hydrodynamic fields for the current time step
        # currently not in use. 
        #update_hydrodynamics!(state, grid, time_sim )

        # 2. Solve horizontal advection and diffusion
        horizontal_transport!(state, grid, dt)

        # 3. Solve vertical advection and diffusion
        vertical_transport!(state, grid, dt)

        # 4. Solve source/sink terms (reactions)
        # This is where previrS package will be integrated.
        source_sink_terms!(state, grid, dt)
    end
    
    total_wall_time_str = format_elapsed_time(start_wall_time)
    @info "Simulation finished (Total wall time: $(total_wall_time_str))" 
    return state # Return the final state
end

end # module TimeSteppingModule