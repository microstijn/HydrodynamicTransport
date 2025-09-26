# src/SourceSinkModule.jl

module SourceSinkModule

export source_sink_terms!

using ..ModelStructs

"""
    source_sink_terms!(state, grid, sources, time, dt)

Updates tracer concentrations based on reaction kinetics and point sources.
This version evaluates the time-dependent influx rate for each source.
"""
function source_sink_terms!(state::State, grid::Grid, sources::Vector{PointSource}, time::Float64, dt::Float64)
    # Time-Dependent Point Source Influx
    for source in sources
        if haskey(state.tracers, source.tracer_name)
            C = state.tracers[source.tracer_name]
            i, j, k = source.i, source.j, source.k
            
            if checkbounds(Bool, C, i, j, k)
                source_volume = grid.volume[i, j, k]
                if source_volume > 0
                    # Evaluate the source's influx rate at the current time
                    current_influx_rate = source.influx_rate(time)
                    C[i, j, k] += (current_influx_rate * dt) / source_volume
                end
            end
        end
    end

    # Existing Decay Logic (unchanged) 
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        decay_rate = 0.1 / (24 * 3600)
        for idx in eachindex(C_dissolved)
            if grid.mask[idx]; C_dissolved[idx] *= (1 - decay_rate * dt); end
        end
    end

    if haskey(state.tracers, :C_sorbed)
        C_sorbed = state.tracers[:C_sorbed]
        decay_rate = 0.1 / (24 * 3600)
        for idx in eachindex(C_sorbed)
            if grid.mask[idx]; C_sorbed[idx] *= (1 - decay_rate * dt); end
        end
    end

    return nothing
end

end # module SourceSinkModule
