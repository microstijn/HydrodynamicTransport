# src/SourceSinkModule.jl

module SourceSinkModule

export source_sink_terms!

using ..ModelStructs

function source_sink_terms!(state::State, grid::CartesianGrid, sources::Vector{PointSource}, time::Float64, dt::Float64)
    # Point Source Influx
    for source in sources
        if haskey(state.tracers, source.tracer_name)
            C = state.tracers[source.tracer_name]
            i, j, k = source.i, source.j, source.k
            if checkbounds(Bool, C, i, j, k)
                current_influx_rate = source.influx_rate(time)
                C[i, j, k] += (current_influx_rate * dt) / grid.volume[i, j, k]
            end
        end
    end

    # Decay Logic
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        decay_rate = 0.1 / (24 * 3600)
        for i in eachindex(C_dissolved)
            if grid.mask[i]; C_dissolved[i] *= (1 - decay_rate * dt); end
        end
    end
end

function source_sink_terms!(state::State, grid::CurvilinearGrid, sources::Vector{PointSource}, time::Float64, dt::Float64)
    # Point Source Influx
    for source in sources
        if haskey(state.tracers, source.tracer_name)
            C = state.tracers[source.tracer_name]
            i, j, k = source.i, source.j, source.k
            if checkbounds(Bool, C, i, j, k)
                current_influx_rate = source.influx_rate(time)
                C[i, j, k] += (current_influx_rate * dt) / grid.volume[i, j, k]
            end
        end
    end
    
    # Decay Logic
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        decay_rate = 0.1 / (24 * 3600)
        
        # Use the 2D rho_mask broadcasted across the vertical dimension
        for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
            if grid.mask_rho[i, j]; C_dissolved[i, j, k] *= (1 - decay_rate * dt); end
        end
    end
end

end # module SourceSinkModule