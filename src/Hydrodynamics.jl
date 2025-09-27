# src/Hydrodynamics.jl

module HydrodynamicsModule

export update_hydrodynamics!, update_hydrodynamics_placeholder!

using ..ModelStructs
using NCDatasets

function find_time_index(time_dim, current_time)
    _, index = findmin(abs.(time_dim .- current_time))
    return index
end

# The placeholder function for testing is unchanged...
function update_hydrodynamics_placeholder!(state::State, grid::Grid, time::Float64)
    nx, ny, _ = grid.dims
    dx = grid.x[2, 1, 1] - grid.x[1, 1, 1]; dy = grid.y[1, 2, 1] - grid.y[1, 1, 1]
    Lx = nx * dx; Ly = ny * dy
    center_x = Lx / 2; center_y = Ly / 2
    max_speed = 0.2; decay_radius = Lx / 5
    u_centered = zeros(size(grid.x)); v_centered = zeros(size(grid.y))
    for k in 1:grid.dims[3], j in 1:ny, i in 1:nx
        rx = grid.x[i, j, k] - center_x; ry = grid.y[i, j, k] - center_y
        r = sqrt(rx^2 + ry^2)
        speed = max_speed * (r / decay_radius) * exp(-0.5 * (r / decay_radius)^2)
        if r > 0; u_centered[i, j, k] = -speed * (ry / r); v_centered[i, j, k] =  speed * (rx / r); end
    end
    for k in 1:grid.dims[3], j in 1:ny, i in 2:nx; state.u[i, j, k] = 0.5 * (u_centered[i-1, j, k] + u_centered[i, j, k]); end
    for k in 1:grid.dims[3], i in 1:nx, j in 2:ny; state.v[i, j, k] = 0.5 * (v_centered[i, j-1, k] + v_centered[i, j, k]); end
    state.w .= 0.0
    return nothing
end


"""
    update_hydrodynamics!(state::State, grid::Grid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64)

Updates hydrodynamic fields by reading 2D or 3D data from a NetCDF file.
"""
function update_hydrodynamics!(state::State, grid::Grid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64)
    time_var_name = hydro_data.var_map[:time]
    time_dim_datetime = ds[time_var_name][:]
    t0 = time_dim_datetime[1]
    time_dim_seconds = [(dt - t0).value / 1000.0 for dt in time_dim_datetime]
    time_idx = find_time_index(time_dim_seconds, time)
    
    nx, ny, nz = grid.dims
    
    if nz == 1
        # 2D Case: Read the surface layer only
        u_slice = ds[hydro_data.var_map[:u]][:, :, end, time_idx]
        state.u[2:nx, :, 1] .= coalesce.(u_slice, 0.0)
        
        v_slice = ds[hydro_data.var_map[:v]][:, :, end, time_idx]
        state.v[:, 2:ny, 1] .= coalesce.(v_slice, 0.0)
        
        temp_slice = ds[hydro_data.var_map[:temp]][:, :, end, time_idx]
        state.temperature[:, :, 1] .= coalesce.(temp_slice, 0.0)

        salt_slice = ds[hydro_data.var_map[:salt]][:, :, end, time_idx]
        state.salinity[:, :, 1] .= coalesce.(salt_slice, 0.0)
    else
        # --- 3D Case: Read the full water column ---
        u_data = ds[hydro_data.var_map[:u]][:, :, :, time_idx]
        state.u[2:nx, :, :] .= coalesce.(u_data, 0.0)

        v_data = ds[hydro_data.var_map[:v]][:, :, :, time_idx]
        state.v[:, 2:ny, :] .= coalesce.(v_data, 0.0)

        temp_data = ds[hydro_data.var_map[:temp]][:, :, :, time_idx]
        state.temperature .= coalesce.(temp_data, 0.0)
        
        salt_data = ds[hydro_data.var_map[:salt]][:, :, :, time_idx]
        state.salinity .= coalesce.(salt_data, 0.0)
    end
    
    state.w .= 0.0
    return nothing
end

end # module HydrodynamicsModule