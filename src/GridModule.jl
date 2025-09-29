# src/GridModule.jl

module GridModule

export initialize_cartesian_grid, initialize_curvilinear_grid,
       interpolate_center_to_xface!, interpolate_center_to_yface!

using ..ModelStructs
using StaticArrays
using NCDatasets

# (initialize_cartesian_grid and initialize_curvilinear_grid are unchanged)
"
    initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz) -> CartesianGrid
"
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
        trues(nx, ny, nz)
    )
    return grid
end

"
    initialize_curvilinear_grid(netcdf_filepath::String) -> CurvilinearGrid
"
function initialize_curvilinear_grid(netcdf_filepath::String)::CurvilinearGrid
    ds = NCDataset(netcdf_filepath)

    lon_rho = ds["lon_rho"][:,:]; lat_rho = ds["lat_rho"][:,:]
    lon_u = ds["lon_u"][:,:]; lat_u = ds["lat_u"][:,:]
    lon_v = ds["lon_v"][:,:]; lat_v = ds["lat_v"][:,:]
    pm = ds["pm"][:,:]; pn = ds["pn"][:,:]
    angle = ds["angle"][:,:]; h = ds["h"][:,:]
    mask_rho = ds["mask_rho"][:,:] .== 1
    mask_u = ds["mask_u"][:,:] .== 1
    mask_v = ds["mask_v"][:,:] .== 1
    
    nx, ny = size(lon_rho)
    z_w = ds["z_w"][:]
    nz = length(z_w) - 1

    dx = 1 ./ pm; dy = 1 ./ pn
    dz = z_w[2:end] - z_w[1:end-1]

    face_area_x = zeros(nx + 1, ny, nz)
    face_area_y = zeros(nx, ny + 1, nz)
    volume = zeros(nx, ny, nz)

    for k in 1:nz
        for j in 1:ny, i in 1:nx
            volume[i,j,k] = dx[i,j] * dy[i,j] * dz[k]
        end
        for j in 1:ny, i in 1:nx+1
            local_dy = i > 1 && i <= nx ? 0.5 * (dy[i-1, j] + dy[i, j]) : dy[min(i,nx), j]
            face_area_x[i,j,k] = local_dy * dz[k]
        end
        for j in 1:ny+1, i in 1:nx
            local_dx = j > 1 && j <= ny ? 0.5 * (dx[i, j-1] + dx[i, j]) : dx[i, min(j,ny)]
            face_area_y[i,j,k] = local_dx * dz[k]
        end
    end
    
    close(ds)

    return CurvilinearGrid(nx, ny, nz, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v, 
                           z_w, pm, pn, angle, h, mask_rho, mask_u, mask_v, 
                           face_area_x, face_area_y, volume)
end


function interpolate_center_to_xface!(xface_field::Array{Float64, 3}, center_field::Array{Float64, 3}, grid::CartesianGrid)
    nx, ny, nz = grid.dims
    for k in 1:nz, j in 1:ny, i in 2:nx
        xface_field[i, j, k] = 0.5 * (center_field[i-1, j, k] + center_field[i, j, k])
    end
    for k in 1:nz, j in 1:ny
        xface_field[1, j, k] = center_field[1, j, k]
        xface_field[nx+1, j, k] = center_field[nx, j, k]
    end
end

function interpolate_center_to_xface!(xface_field::Array{Float64, 3}, center_field::Array{Float64, 3}, grid::CurvilinearGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for k in 1:nz, j in 1:ny, i in 2:nx
        dx_left = 1 / (2 * grid.pm[i-1, j]); dx_right = 1 / (2 * grid.pm[i, j])
        total_dx = dx_left + dx_right
        xface_field[i, j, k] = (center_field[i-1, j, k] * dx_right + center_field[i, j, k] * dx_left) / total_dx
    end
    for k in 1:nz, j in 1:ny; xface_field[1, j, k] = center_field[1, j, k]; xface_field[nx+1, j, k] = center_field[nx, j, k]; end
end

function interpolate_center_to_yface!(yface_field::Array{Float64, 3}, center_field::Array{Float64, 3}, grid::CartesianGrid)
    nx, ny, nz = grid.dims
    for k in 1:nz, j in 2:ny, i in 1:nx; yface_field[i, j, k] = 0.5 * (center_field[i, j-1, k] + center_field[i, j, k]); end
    for k in 1:nz, i in 1:nx; yface_field[i, 1, k] = center_field[i, 1, k]; yface_field[i, ny+1, k] = center_field[i, ny, k]; end
end

function interpolate_center_to_yface!(yface_field::Array{Float64, 3}, center_field::Array{Float64, 3}, grid::CurvilinearGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for k in 1:nz, j in 2:ny, i in 1:nx
        dy_bottom = 1 / (2 * grid.pn[i, j-1]); dy_top = 1 / (2 * grid.pn[i, j])
        total_dy = dy_bottom + dy_top
        yface_field[i, j, k] = (center_field[i, j-1, k] * dy_top + center_field[i, j, k] * dy_bottom) / total_dy
    end
    for k in 1:nz, i in 1:nx; yface_field[i, 1, k] = center_field[i, 1, k]; yface_field[i, ny+1, k] = center_field[i, ny, k]; end
end

end # module GridModule