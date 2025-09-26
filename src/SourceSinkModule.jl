# src/SourceSinkTerms.jl

module SourceSinkModule

export source_sink_terms!

using ..ModelStructs

"""
    source_sink_terms!(state::State, grid::Grid, dt::Float64)

Updates tracer concentrations based on reaction kinetics (sources and sinks).

This function iterates through every active grid cell and applies the biogeochemical
reaction model for a single time step. It is the integration point for the
`previrS` virus fate logic.
"""
function source_sink_terms!(state::State, grid::Grid, dt::Float64)
    nx, ny, nz = grid.dims

    # Check if the required tracers exist. If not, do nothing.
    if !haskey(state.tracers, :C_dissolved) || !haskey(state.tracers, :C_sorbed)
        return nothing
    end

    C_dissolved = state.tracers[:C_dissolved]
    C_sorbed = state.tracers[:C_sorbed]

    # Loop over every single grid cell in the domain
    for k in 1:nz, j in 1:ny, i in 1:nx
        if grid.mask[i,j,k]
            # 1. Gather inputs for this specific cell (i,j,k)
            # 2. Call the previrS ODE solver for a single time step `dt`
            # 3. Update the state with the results
            # placeholder: a simple first-order decay
            decay_rate = 0.1 / (24 * 3600) # 10% decay per day, converted to per second
            C_dissolved[i,j,k] *= (1 - decay_rate * dt)
            C_sorbed[i,j,k] *= (1 - decay_rate * dt)
        end
    end

    return nothing
end

end # module SourceSinkModule