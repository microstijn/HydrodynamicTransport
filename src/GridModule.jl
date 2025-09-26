# src/GridModule.jl

module GridModule

export initialize_grid, interpolate_center_to_xface!, interpolate_center_to_yface!

using ..ModelStructs
using StaticArrays

"""
    initialize_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)

Constructs a `Grid` object for a uniform, rectangular computational domain.
This version correctly sizes the face_area arrays for a staggered Arakawa C-grid.
"""
function initialize_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # Cell-centered coordinates
    x_centers = (1:nx) .* dx .- (dx / 2)
    y_centers = (1:ny) .* dy .- (dy / 2)
    z_centers = (1:nz) .* dz .- (dz / 2)
    x_3d = [x for x in x_centers, y in y_centers, z in z_centers]
    y_3d = [y for x in x_centers, y in y_centers, z in z_centers]
    z_3d = [z for x in x_centers, y in y_centers, z in z_centers]

    # ARAKAWA C-GRID MODIFICATION
    # Face areas must have staggered dimensions to match their corresponding velocities.
    cell_volume = dx * dy * dz
    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy

    grid = Grid(
        SVector(nx, ny, nz),
        x_3d,
        y_3d,
        z_3d,
        fill(cell_volume, (nx, ny, nz)),
        fill(area_x, (nx + 1, ny, nz)), # Staggered dimension for U-faces
        fill(area_y, (nx, ny + 1, nz)), # Staggered dimension for V-faces
        fill(area_z, (nx, ny, nz + 1)), # Staggered dimension for W-faces
        fill(true, (nx, ny, nz))
    )

    return grid
end

# ARAKAWA C-GRID MODIFICATION: ADD INTERPOLATION FUNCTIONS

"""
    interpolate_center_to_xface!(center_field, xface_field)

Interpolates a scalar field from cell centers to x-faces (U-points) in-place.
This is a simple linear average.
"""
function interpolate_center_to_xface!(center_field::Array{Float64, 3}, xface_field::Array{Float64, 3})
    nx, ny, nz = size(center_field)
    
    # Interior faces
    for k in 1:nz, j in 1:ny, i in 2:nx
        xface_field[i, j, k] = 0.5 * (center_field[i-1, j, k] + center_field[i, j, k])
    end
    
    # Boundary faces (simple extrapolation: use the value from the adjacent center)
    for k in 1:nz, j in 1:ny
        xface_field[1, j, k] = center_field[1, j, k]
        xface_field[nx+1, j, k] = center_field[nx, j, k]
    end
    return nothing
end

"""
    interpolate_center_to_yface!(center_field, yface_field)

Interpolates a scalar field from cell centers to y-faces (V-points) in-place.
"""
function interpolate_center_to_yface!(center_field::Array{Float64, 3}, yface_field::Array{Float64, 3})
    nx, ny, nz = size(center_field)

    # Interior faces
    for k in 1:nz, j in 2:ny, i in 1:nx
        yface_field[i, j, k] = 0.5 * (center_field[i, j-1, k] + center_field[i, j, k])
    end

    # Boundary faces
    for k in 1:nz, i in 1:nx
        yface_field[i, 1, k] = center_field[i, 1, k]
        yface_field[i, ny+1, k] = center_field[i, ny, k]
    end
    return nothing
end

end # module GridModule