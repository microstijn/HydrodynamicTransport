# src/SourceSinkModule.jl

module SourceSinkModule

export source_sink_terms!

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.ModelStructs: CurvilinearGrid, CartesianGrid

"""
    _find_nearest_wet_neighbor(start_i, start_j, grid, state, D_crit)

Performs an expanding box search to find the nearest "wet" cell.

A cell is considered "wet" if it's a water cell (`mask_rho` is true) and its total
water depth (`h + zeta`) exceeds the critical depth `D_crit`.
"""
function _find_nearest_wet_neighbor(start_i::Int, start_j::Int, grid::CurvilinearGrid, state::State, D_crit::Float64)
    ng = grid.ng
    nx_phys, ny_phys = grid.nx, grid.ny

    # Search in an expanding box around the starting point
    for radius in 1:max(nx_phys, ny_phys)
        for i_offset in -radius:radius, j_offset in -radius:radius
            if abs(i_offset) < radius && abs(j_offset) < radius
                continue
            end

            i_candidate = start_i + i_offset
            j_candidate = start_j + j_offset

            if 1 <= i_candidate <= nx_phys && 1 <= j_candidate <= ny_phys
                i_glob = i_candidate + ng
                j_glob = j_candidate + ng
                
                # Check if the cell is a water cell and if total depth is sufficient
                if grid.mask_rho[i_glob, j_glob] && (grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, grid.nz] > D_crit)
                    return (i_candidate, j_candidate)
                end
            end
        end
    end

    @warn "Could not find a wet neighbor for source at ($start_i, $start_j). Using original location."
    return (start_i, start_j) # Fallback
end

function source_sink_terms!(state::State, grid::AbstractGrid, sources::Vector{PointSource}, time::Float64, dt::Float64, D_crit::Float64)
    ng = grid.ng

    # --- Point Source Influx ---
    # The logic is now split based on grid type to accommodate the relocation feature
    # for CurvilinearGrids while preserving original behavior for others.
    if grid isa CurvilinearGrid
        nx_phys, ny_phys, nz_phys = grid.nx, grid.ny, grid.nz
        for source in sources
            if haskey(state.tracers, source.tracer_name)
                C = state.tracers[source.tracer_name]
                target_i, target_j = source.i, source.j

                if source.relocate_if_dry
                    i_glob_orig = source.i + ng
                    j_glob_orig = source.j + ng
                    
                    is_dry = !grid.mask_rho[i_glob_orig, j_glob_orig] || 
                             (grid.h[i_glob_orig, j_glob_orig] + state.zeta[i_glob_orig, j_glob_orig, nz_phys] <= D_crit)

                    if is_dry
                        target_i, target_j = _find_nearest_wet_neighbor(source.i, source.j, grid, state, D_crit)
                    end
                end
                
                target_k = (target_i != source.i || target_j != source.j) ? nz_phys : source.k
                i_glob, j_glob, k_glob = target_i + ng, target_j + ng, target_k
                
                if checkbounds(Bool, C, i_glob, j_glob, k_glob)
                    if grid.volume[i_glob, j_glob, k_glob] > 1e-12
                        current_influx_rate = source.influx_rate(time)
                        C[i_glob, j_glob, k_glob] += (current_influx_rate * dt) / grid.volume[i_glob, j_glob, k_glob]
                    else
                         @warn "Target cell volume is zero at physical index ($target_i, $target_j, $target_k). Source ignored."
                    end
                else
                    @warn "Source target at physical index ($target_i, $target_j, $target_k) is out of bounds."
                end
            end
        end
    else # For CartesianGrid and any other AbstractGrid types
        for source in sources
            if haskey(state.tracers, source.tracer_name)
                C = state.tracers[source.tracer_name]
                i_glob, j_glob, k_glob = source.i + ng, source.j + ng, source.k
                if checkbounds(Bool, C, i_glob, j_glob, k_glob)
                    current_influx_rate = source.influx_rate(time)
                    C[i_glob, j_glob, k_glob] += (current_influx_rate * dt) / grid.volume[i_glob, j_glob, k_glob]
                end
            end
        end
    end

    # --- Decay Logic (Operates on Physical Domain for ALL grid types) ---
    if haskey(state.tracers, :C_dissolved)
        C_dissolved = state.tracers[:C_dissolved]
        decay_rate = 0.1 / (24 * 3600)
        
        nx_p, ny_p, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
        
        for k in axes(C_dissolved, 3), j_phys in 1:ny_p, i_phys in 1:nx_p
            i_glob = i_phys + ng
            j_glob = j_phys + ng

            # Use the appropriate mask based on grid type
            mask_to_use = isa(grid, CurvilinearGrid) ? grid.mask_rho[i_glob, j_glob] : grid.mask[i_glob, j_glob, k]
            if mask_to_use
                C_dissolved[i_glob, j_glob, k] *= (1 - decay_rate * dt)
            end
        end
    end
end

end # module SourceSinkModule