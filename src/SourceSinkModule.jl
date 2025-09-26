# src/SourceSinkTerms.jl

module SourceSinkModule

export source_sink_terms!

using ..ModelStructs

"""
    source_sink_terms!(state::State, grid::Grid, dt::Float64)

Updates tracer concentrations based on reaction kinetics (sources and sinks).

This function iterates through every active grid cell and applies the biogeochemical
reaction model for a single time step. It flexibly applies logic to any
tracers it knows how to handle.
"""
function source_sink_terms!(state::State, grid::Grid, dt::Float64)
    # We now check for each tracer individually and apply its logic if it exists.
    # This makes the module robust and independent of which tracers are in the simulation.

    # Apply decay to the dissolved tracer if it's part of the state
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        # placeholder: a simple first-order decay
        decay_rate = 0.1 / (24 * 3600) # 10% decay per day, converted to per second

        for i in eachindex(C_dissolved)
            if grid.mask[i]
                C_dissolved[i] *= (1 - decay_rate * dt)
            end
        end
    end

    # Apply decay to the sorbed tracer if it's part of the state
    if haskey(state.tracers, :C_sorbed)
        C_sorbed = state.tracers[:C_sorbed]
        decay_rate = 0.1 / (24 * 3600) # (can be different for different tracers)
        
        for i in eachindex(C_sorbed)
            if grid.mask[i]
                C_sorbed[i] *= (1 - decay_rate * dt)
            end
        end
    end

    # Add other reaction kinetics for other tracers here...

    return nothing
end

end # module SourceSinkModule