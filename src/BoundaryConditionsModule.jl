# src/BoundaryConditionsModule.jl

module BoundaryConditionsModule

export apply_boundary_conditions!

using ..HydrodynamicTransport.ModelStructs

function apply_boundary_conditions!(state::State, grid::AbstractGrid, bcs::Vector{<:BoundaryCondition})
    ng = grid.ng
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

    # --- Create helpers to find the most specific BC for a given side ---
    find_river_bc(side, i) = findfirst(b -> isa(b, RiverBoundary) && b.side == side && i in b.indices, bcs)
    find_tidal_bc(side) = findfirst(b -> isa(b, TidalBoundary) && b.side == side, bcs)
    find_open_bc(side) = findfirst(b -> isa(b, OpenBoundary) && b.side == side, bcs)

    # --- Pass 1: Process East-West sides ---
    # This pass iterates over the physical y-dimension of the relevant arrays.
    for k in axes(state.u, 3), j_phys in 1:ny
        j_glob = j_phys + ng
        
        # --- WEST ---
        river_bc_west_idx = find_river_bc(:West, j_phys)
        open_bc_west_idx = find_open_bc(:West)
        
        if river_bc_west_idx !== nothing
            bc = bcs[river_bc_west_idx]
            state.tracers[bc.tracer_name][1:ng, j_glob, k] .= bc.concentration(state.time)
            state.u[ng+1, j_glob, k] = bc.velocity(state.time)
        elseif open_bc_west_idx !== nothing
            boundary_velocity = state.u[ng+1, j_glob, k]
            if boundary_velocity <= 0 # Outflow
                for tracer in values(state.tracers); tracer[1:ng, j_glob, k] .= tracer[ng+1, j_glob, k]; end
            else # Inflow
                for tracer in values(state.tracers); tracer[1:ng, j_glob, k] .= 0.0; end
            end
        end

        # --- EAST ---
        tidal_bc_east_idx = find_tidal_bc(:East)
        open_bc_east_idx = find_open_bc(:East)

        if tidal_bc_east_idx !== nothing
            bc = bcs[tidal_bc_east_idx]
            inflow_values = bc.inflow_concentrations(state.time)
            boundary_velocity = state.u[nx+ng+1, j_glob, k]
            if boundary_velocity < 0 # Inflow
                # --- FIX: Use a safe default of 0.0 for unspecified tracers during inflow ---
                for (name, arr) in state.tracers; arr[nx+ng+1:nx_tot, j_glob, k] .= get(inflow_values, name, 0.0); end
            else # Outflow
                for tracer in values(state.tracers); tracer[nx+ng+1:nx_tot, j_glob, k] .= tracer[nx+ng, j_glob, k]; end
            end
        elseif open_bc_east_idx !== nothing
            boundary_velocity = state.u[nx+ng+1, j_glob, k]
            if boundary_velocity >= 0 # Outflow
                for tracer in values(state.tracers); tracer[nx+ng+1:nx_tot, j_glob, k] .= tracer[nx+ng, j_glob, k]; end
            else # Inflow
                for tracer in values(state.tracers); tracer[nx+ng+1:nx_tot, j_glob, k] .= 0.0; end
            end
        end
    end

    # --- Pass 2: Process North-South sides (this will correctly fill corners) ---
    # This pass iterates over the FULL width of the arrays (i_glob in 1:nx_tot).
    for k in axes(state.v, 3), i_glob in 1:nx_tot
        i_phys = i_glob - ng # Physical index can be outside [1,nx] here, that's okay

        # --- SOUTH ---
        # Note: river logic for N/S boundaries would need i_phys
        open_bc_south_idx = find_open_bc(:South) 
        if open_bc_south_idx !== nothing
            boundary_velocity = state.v[i_glob, ng+1, k]
            if boundary_velocity <= 0 # Outflow
                for tracer in values(state.tracers); tracer[i_glob, 1:ng, k] .= tracer[i_glob, ng+1, k]; end
            else # Inflow
                for tracer in values(state.tracers); tracer[i_glob, 1:ng, k] .= 0.0; end
            end
        end
        
        # --- NORTH ---
        open_bc_north_idx = find_open_bc(:North)
        if open_bc_north_idx !== nothing
            boundary_velocity = state.v[i_glob, ny+ng+1, k]
            if boundary_velocity < 0 # Inflow
                for tracer in values(state.tracers); tracer[i_glob, ny+ng+1:ny_tot, k] .= 0.0; end
            else # Outflow
                for tracer in values(state.tracers); tracer[i_glob, ny+ng+1:ny_tot, k] .= tracer[i_glob, ny+ng, k]; end
            end
        end
    end
end

end # module BoundaryConditionsModule