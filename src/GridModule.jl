# src/Grid.jl

module GridModule

export initialize_grid

# We use 'using ..ModelStructs' to access the structs defined
# in the sibling file 'Structs.jl'. The '..' goes up one
# level to the parent HydrodynamicTransport module.
using ..ModelStructs
using StaticArrays

"""
    initialize_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)

Constructs a `Grid` object for a uniform, rectangular computational domain.

The function creates the geometric foundation for an Arakawa 'C' grid, where scalar
quantities (like tracers) are located at the cell centers. The domain is assumed to
originate at `(0,0,0)`.

# Arguments
- `nx::Int`: Number of active grid cells in the x-dimension.
- `ny::Int`: Number of active grid cells in the y-dimension.
- `nz::Int`: Number of active grid cells in the z-dimension.
- `Lx::Float64`: The total physical length of the domain in the x-dimension (meters).
- `Ly::Float64`: The total physical length of the domain in the y-dimension (meters).
- `Lz::Float64`: The total physical depth of the domain in the z-dimension (meters).

# Returns
- A fully populated `Grid` struct. The coordinates (`x`, `y`, `z`) represent the
  center of each grid cell. The `mask` field is initialized to `true` for all
  cells, indicating an entirely wet domain.

# Examples
```jldoctest
julia> grid = initialize_grid(10, 20, 5, 100.0, 200.0, 50.0);

julia> grid.dims
3-element SVector{3, Int64} with indices SOneTo(3):
 10
 20
  5

julia> dx = 100.0 / 10; dy = 200.0 / 20; dz = 50.0 / 5;

julia> grid.volume[1, 1, 1] == dx * dy * dz
true

julia> grid.x[1, 1, 1] # Center of the first cell in x
5.0
"""
function initialize_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    x_centers = (1:nx) .* dx .- (dx / 2)
    y_centers = (1:ny) .* dy .- (dy / 2)
    z_centers = (1:nz) .* dz .- (dz / 2)

    x_3d = [x for x in x_centers, y in y_centers, z in z_centers]
    y_3d = [y for x in x_centers, y in y_centers, z in z_centers]
    z_3d = [z for x in x_centers, y in y_centers, z in z_centers]

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
        fill(area_x, (nx, ny, nz)),
        fill(area_y, (nx, ny, nz)),
        fill(area_z, (nx, ny, nz)),
        fill(true, (nx, ny, nz))
    )

    return grid
end

end # module GridModule