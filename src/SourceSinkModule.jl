# src/SourceSinkModule.jl

module SourceSinkModule

export source_sink_terms!

using ..HydrodynamicTransport.ModelStructs

function source_sink_terms!(state::State, grid::AbstractGrid, sources::Vector{PointSource}, time::Float64, dt::Float64)
    ng = grid.ng
    nx_phys, ny_phys, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)

    # --- Point Source Influx (Ghost-Cell Aware) ---
    for source in sources
        if haskey(state.tracers, source.tracer_name)
            C = state.tracers[source.tracer_name]
            
            # Translate user-provided physical indices to global array indices
            i_glob = source.i + ng
            j_glob = source.j + ng
            k_glob = source.k # No ghost cells in vertical direction
            
            # Check bounds on the full array
            if checkbounds(Bool, C, i_glob, j_glob, k_glob)
                current_influx_rate = source.influx_rate(time)
                # Use global indices to update the concentration and read volume
                C[i_glob, j_glob, k_glob] += (current_influx_rate * dt) / grid.volume[i_glob, j_glob, k_glob]
            else
                @warn "Source at physical index ($(source.i), $(source.j), $(source.k)) is out of bounds."
            end
        end
    end

    # --- Decay Logic (Operates on Physical Domain Only) ---
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        decay_rate = 0.1 / (24 * 3600)
        
        # Loop over only the physical interior of the domain
        for k in axes(C_dissolved, 3), j_phys in 1:ny_phys, i_phys in 1:nx_phys
            i_glob = i_phys + ng
            j_glob = j_phys + ng

            # The mask is also defined on the full grid, so use global indices
            if grid.mask[i_glob, j_glob, k]
                C_dissolved[i_glob, j_glob, k] *= (1 - decay_rate * dt)
            end
        end
    end
end

end # module SourceSinkModule

