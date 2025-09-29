# src/GridModule.jl

module GridModule

export initialize_cartesian_grid, initialize_curvilinear_grid,
       interpolate_center_to_xface!, interpolate_center_to_yface!

using ..ModelStructs
using StaticArrays
using NCDatasets

"""
    initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz) -> CartesianGrid
"""
function initialize_cartesian_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64)::CartesianGrid
    dx = Lx / nx; dy = Ly / ny; dz = Lz / nz
    x_centers = (1:nx) .* dx .- (dx / 2); y_centers = (1:ny) .* dy .- (dy / 2); z_centers = (1:nz) .* dz .- (dz / 2)
    x_3d = [x for x in x_centers, y in y_centers, z in z_centers]
    y_3d = [y for x in x_centers, y in y_centers, z in z_centers]
    z_3d = [z for x in x_centers, y in y_centers, z in z_centers]
    
    grid = CartesianGrid(
        SVector(nx, ny, nz), x_3d, y_3d, z_3d,
        fill(dx * dy * dz, (nx, ny, nz)),
        fill(dy * dz, (nx + 1, ny, nz)),
        fill(dx * dz, (nx, ny + 1, nz)),
        fill(dx * dy, (nx, ny, nz + 1)),
        fill(true, (nx, ny, nz))
    )
    return grid
end


"""
    initialize_curvilinear_grid(filepath::String) -> CurvilinearGrid
"""
function initialize_curvilinear_grid(filepath::String)::CurvilinearGrid
    NCDataset(filepath, "r") do ds
        # Read Dimensions
        nx = ds.dim["xi_rho"]; ny = ds.dim["eta_rho"]; nz = ds.dim["s_rho"]
        
        # Read Primary 2D Metrics and Coordinates
        lon_rho = ds["lon_rho"][:,:]; lat_rho = ds["lat_rho"][:,:]
        lon_u = ds["lon_u"][:,:]; lat_u = ds["lat_u"][:,:]
        lon_v = ds["lon_v"][:,:]; lat_v = ds["lat_v"][:,:]
        pm = ds["pm"][:,:]; pn = ds["pn"][:,:]
        angle = ds["angle"][:,:]; h = ds["h"][:,:]
        mask_rho = Bool.(ds["mask_rho"][:,:]); mask_u = Bool.(ds["mask_u"][:,:]); mask_v = Bool.(ds["mask_v"][:,:])
        
        # Handle Vertical Coordinates (Simplified)
        z_w = ds["s_w"][:] .* maximum(h)
        dz_vec = z_w[2:end] - z_w[1:end-1]
        
        # Calculate Derived 3D Metrics
        volume = zeros(Float64, nx, ny, nz)
        face_area_x = zeros(Float64, nx + 1, ny, nz)
        face_area_y = zeros(Float64, nx, ny + 1, nz)

        # Interpolate 1/pn to get cell heights at U-faces (dy_u)
        dy_u = zeros(Float64, nx + 1, ny)
        for j in 1:ny, i in 2:nx
            dy_u[i, j] = 1 / (0.5 * (pn[i-1, j] + pn[i, j]))
        end
        # --- FIX: Extrapolate for boundary faces ---
        dy_u[1, :] .= dy_u[2, :]
        dy_u[nx+1, :] .= dy_u[nx, :]

        # Interpolate 1/pm to get cell widths at V-faces (dx_v)
        dx_v = zeros(Float64, nx, ny + 1)
        for j in 2:ny, i in 1:nx
            dx_v[i, j] = 1 / (0.5 * (pm[i, j-1] + pm[i, j]))
        end
        # --- FIX: Extrapolate for boundary faces ---
        dx_v[:, 1] .= dx_v[:, 2]
        dx_v[:, ny+1] .= dx_v[:, ny]
        
        # Populate the 3D metric arrays
        for k in 1:nz
            dz = dz_vec[k]
            volume[:, :, k] = (1 ./ (pm .* pn)) .* dz
            face_area_x[:, :, k] = dy_u .* dz
            face_area_y[:, :, k] = dx_v .* dz
        end
        
        return CurvilinearGrid(
            nx, ny, nz, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
            z_w, pm, pn, angle, h, mask_rho, mask_u, mask_v,
            face_area_x, face_area_y, volume
        )
    end
end

# Interpolation functions are unchanged...
function interpolate_center_to_xface!(center_field::Array{Float64, 3}, xface_field::Array{Float64, 3}, grid::CartesianGrid)
    nx, ny, nz = grid.dims
    for k in 1:nz, j in 1:ny, i in 2:nx; xface_field[i, j, k] = 0.5 * (center_field[i-1, j, k] + center_field[i, j, k]); end
    for k in 1:nz, j in 1:ny; xface_field[1, j, k] = center_field[1, j, k]; xface_field[nx+1, j, k] = center_field[nx, j, k]; end
end

function interpolate_center_to_xface!(center_field::Array{Float64, 3}, xface_field::Array{Float64, 3}, grid::CurvilinearGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for k in 1:nz, j in 1:ny, i in 2:nx
        dx_left = 1 / (2 * grid.pm[i-1, j]); dx_right = 1 / (2 * grid.pm[i, j])
        total_dx = dx_left + dx_right
        xface_field[i, j, k] = (center_field[i-1, j, k] * dx_right + center_field[i, j, k] * dx_left) / total_dx
    end
    for k in 1:nz, j in 1:ny; xface_field[1, j, k] = center_field[1, j, k]; xface_field[nx+1, j, k] = center_field[nx, j, k]; end
end

function interpolate_center_to_yface!(center_field::Array{Float64, 3}, yface_field::Array{Float64, 3}, grid::CartesianGrid)
    nx, ny, nz = grid.dims
    for k in 1:nz, j in 2:ny, i in 1:nx; yface_field[i, j, k] = 0.5 * (center_field[i, j-1, k] + center_field[i, j, k]); end
    for k in 1:nz, i in 1:nx; yface_field[i, 1, k] = center_field[i, 1, k]; yface_field[i, ny+1, k] = center_field[i, ny, k]; end
end

function interpolate_center_to_yface!(center_field::Array{Float64, 3}, yface_field::Array{Float64, 3}, grid::CurvilinearGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for k in 1:nz, j in 2:ny, i in 1:nx
        dy_bottom = 1 / (2 * grid.pn[i, j-1]); dy_top = 1 / (2 * grid.pn[i, j])
        total_dy = dy_bottom + dy_top
        yface_field[i, j, k] = (center_field[i, j-1, k] * dy_top + center_field[i, j, k] * dy_bottom) / total_dy
    end
    for k in 1:nz, i in 1:nx; yface_field[i, 1, k] = center_field[i, 1, k]; yface_field[i, ny+1, k] = center_field[i, ny, k]; end
end


end # module GridModule