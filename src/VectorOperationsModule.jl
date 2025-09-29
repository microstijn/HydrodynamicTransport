# src/VectorOperations.jl

module VectorOperationsModule

export rotate_velocities_to_geographic

using ..ModelStructs

"""
    rotate_velocities_to_geographic(grid::CurvilinearGrid, u_stag::AbstractArray, v_stag::AbstractArray)

Rotates grid-aligned velocities (u, v) to geographic velocities (East, North).

This function follows the standard procedure for curvilinear models:
1. Interpolate the staggered u and v components to the cell centers (rho-points).
2. Apply the 2D rotation matrix using the `angle` metric at each rho-point.

# Arguments
- `grid`: A `CurvilinearGrid` containing the `angle` metric.
- `u_stag`: The 2D or 3D array of grid-aligned u-velocity on the u-faces.
- `v_stag`: The 2D or 3D array of grid-aligned v-velocity on the v-faces.

# Returns
- `u_east`, `v_north`: Tuple of 2D or 3D arrays containing the geographic velocity components at the cell centers.
"""
function rotate_velocities_to_geographic(grid::CurvilinearGrid, u_stag::AbstractArray, v_stag::AbstractArray)
    nx, ny, _ = size(grid.volume) # Use volume to get tracer dimensions

    # Create new arrays for the centered and rotated velocities
    u_rho = zeros(Float64, nx, ny, size(u_stag, 3))
    v_rho = zeros(Float64, nx, ny, size(v_stag, 3))
    u_east = zeros(Float64, nx, ny, size(u_stag, 3))
    v_north = zeros(Float64, nx, ny, size(u_stag, 3))

    # 1. Interpolate staggered velocities to cell centers (rho-points)
    for k in 1:size(u_stag, 3) # Loop over vertical layers
        for j in 1:ny, i in 1:nx
            u_rho[i, j, k] = 0.5 * (u_stag[i, j, k] + u_stag[i+1, j, k])
            v_rho[i, j, k] = 0.5 * (v_stag[i, j, k] + v_stag[i, j+1, k])
        end
    end

    # 2. Apply the rotation at each cell center
    for k in 1:size(u_stag, 3)
        for j in 1:ny, i in 1:nx
            ur = u_rho[i, j, k]
            vr = v_rho[i, j, k]
            ang = grid.angle[i, j]
            
            # Apply rotation formulas
            u_east[i, j, k] = ur * cos(ang) - vr * sin(ang)
            v_north[i, j, k] = vr * cos(ang) + ur * sin(ang)
        end
    end

    return u_east, v_north
end

end # module VectorOperationsModule