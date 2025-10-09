# src/BoundaryConditionsModule.jl

module BoundaryConditionsModule

export apply_boundary_conditions!, apply_intermediate_boundary_conditions!

using ..HydrodynamicTransport.ModelStructs

"""
    apply_boundary_conditions!(state::State, grid::AbstractGrid, bcs::Vector{<:BoundaryCondition})

Applies a set of boundary conditions to the model state by updating values in the ghost cells.

This function modifies the `state` object in-place. It iterates through the domain boundaries
and applies the specified conditions from the `bcs` vector. The logic is structured to
handle different boundary types (`OpenBoundary`, `RiverBoundary`, `TidalBoundary`) and to
correctly manage inflow and outflow conditions.

The process is performed in two passes:
1.  **East-West boundaries**: Updates ghost cells on the west and east sides.
2.  **North-South boundaries**: Updates ghost cells on the north and south sides. This two-pass
    approach ensures that corner cells are handled correctly.

For each boundary, the function determines the appropriate action based on the velocity
normal to that boundary:
-   **Outflow**: A zero-gradient condition is applied, where the ghost cell values are set
    equal to the values in the adjacent physical domain cell.
-   **Inflow**: Ghost cell values are set according to the specific boundary condition rule.
    For example, a `TidalBoundary` might specify time-varying tracer concentrations, while a
    simple `OpenBoundary` might default to a concentration of zero.

# Arguments
- `state::State`: The current state of the model, which will be modified.
- `grid::AbstractGrid`: The computational grid, providing dimensions and ghost cell information.
- `bcs::Vector{<:BoundaryCondition}`: A vector of boundary condition objects to be applied.

# Returns
- `nothing`: The function modifies the `state` argument in-place.
"""
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

"""
    apply_intermediate_boundary_conditions!(C_intermediate, C_final, grid, bcs, tracer_name)

Applies boundary conditions to the intermediate solution `C_intermediate` from the
x-sweep. This is a critical step for the stability and accuracy of the ADI method.

For Dirichlet (or fixed value) boundary conditions, the value of the intermediate
field at a boundary cell is set to the known physical boundary value for the final
solution at the new time step. This prevents the unphysical intermediate variable
from polluting the second (Y-sweep) step of the ADI solver.
"""
function apply_intermediate_boundary_conditions!(C_intermediate::Array{Float64, 3}, C_final::Array{Float64, 3}, grid::AbstractGrid, bcs::Vector{<:BoundaryCondition}, tracer_name::Symbol)
    # This function is a placeholder to be implemented correctly.
    # For now, it's a no-op to avoid errors.
end

end # module BoundaryConditionsModule