# src/Hydrodynamics.jl

module HydrodynamicsModule

export update_hydrodynamics!, update_hydrodynamics_placeholder!

using ..ModelStructs
using NCDatasets
using Dates

function find_time_index(time_dim_seconds::Vector{<:Real}, current_time_seconds::Real)
    _, index = findmin(abs.(time_dim_seconds .- current_time_seconds))
    return index
end

function update_hydrodynamics_placeholder!(state::State, grid::CartesianGrid, time::Float64)
    nx, ny, _ = grid.dims
    dx = grid.x[2, 1, 1] - grid.x[1, 1, 1]; dy = grid.y[1, 2, 1] - grid.y[1, 1, 1]
    Lx = nx * dx; Ly = ny * dy
    center_x = Lx / 2; center_y = Ly / 2

    # Vortex parameters
    period = 200.0
    omega = 2π / period
    
    u_centered = zeros(size(grid.x)); v_centered = zeros(size(grid.y))
    for k in 1:grid.dims[3], j in 1:ny, i in 1:nx
        rx = grid.x[i, j, k] - center_x
        ry = grid.y[i, j, k] - center_y
        
        # Geographic (East/North) velocities for a vortex
        u_east = -omega * ry
        v_north = omega * rx
        u_centered[i, j, k] = u_east
        v_centered[i, j, k] = v_north
    end
    
    # Interpolate from cell centers to staggered faces
    for k in 1:grid.dims[3], j in 1:ny, i in 2:nx
        state.u[i, j, k] = 0.5 * (u_centered[i-1, j, k] + u_centered[i, j, k])
    end
    for k in 1:grid.dims[3], i in 1:nx, j in 2:ny
        state.v[i, j, k] = 0.5 * (v_centered[i, j-1, k] + v_centered[i, j, k])
    end
    state.w .= 0.0
end

function update_hydrodynamics_placeholder!(state::State, grid::CurvilinearGrid, time::Float64)
    # --- NEW IMPLEMENTATION for CurvilinearGrid vortex ---
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Use the mean of the coordinate ranges to find the center
    center_x = (maximum(grid.lon_rho) + minimum(grid.lon_rho)) / 2
    center_y = (maximum(grid.lat_rho) + minimum(grid.lat_rho)) / 2
    
    # Vortex parameters
    period = 200.0
    omega = 2π / period
    
    u_centered = zeros(Float64, nx, ny, nz)
    v_centered = zeros(Float64, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        # Geographic (East/North) velocities based on lon/lat coordinates
        rx = grid.lon_rho[i, j] - center_x
        ry = grid.lat_rho[i, j] - center_y
        u_east = -omega * ry
        v_north = omega * rx

        # Rotate from geographic to grid-aligned using the grid angle
        ang = grid.angle[i, j]
        u_centered[i, j, k] = u_east * cos(-ang) - v_north * sin(-ang)
        v_centered[i, j, k] = v_north * cos(-ang) + u_east * sin(-ang)
    end

    # Interpolate from cell centers to staggered faces
    for k in 1:nz, j in 1:ny, i in 2:nx
        state.u[i, j, k] = 0.5 * (u_centered[i-1, j, k] + u_centered[i, j, k])
    end
    for k in 1:nz, i in 1:nx, j in 2:ny
        state.v[i, j, k] = 0.5 * (v_centered[i, j-1, k] + v_centered[i, j, k])
    end
    state.w .= 0.0
end


# --- Real Data Hydrodynamics ---
function update_hydrodynamics!(state::State, grid::AbstractGrid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64)
    time_var_name = get(hydro_data.var_map, :time, "time")
    time_dim_raw = ds[time_var_name][:]
    
    if eltype(time_dim_raw) <: DateTime
        t0 = time_dim_raw[1]
        time_dim_seconds = [(dt - t0).value / 1000.0 for dt in time_dim_raw]
    else
        time_dim_seconds = time_dim_raw
    end
    
    time_idx = find_time_index(time_dim_seconds, time)
    
    nz = isa(grid, CartesianGrid) ? grid.dims[3] : grid.nz

    fields_to_load = Dict(
        state.u => :u, state.v => :v, state.w => :w,
        state.temperature => :temp, state.salinity => :salt
    )

    for (state_field, standard_name) in fields_to_load
        if haskey(hydro_data.var_map, standard_name)
            nc_var_name = hydro_data.var_map[standard_name]
            if haskey(ds, nc_var_name)
                if nz == 1
                    data_slice = ds[nc_var_name][:, :, end, time_idx]
                    target_field = view(state_field, :, :, 1)
                    if size(target_field) == size(data_slice)
                        target_field .= coalesce.(data_slice, 0.0)
                    end
                else
                    data_slice = ds[nc_var_name][:, :, :, time_idx]
                    if size(state_field) == size(data_slice)
                        state_field .= coalesce.(data_slice, 0.0)
                    end
                end
            else
                @warn "Variable '$(nc_var_name)' not found in NetCDF file. Skipping."
            end
        end
    end
    return nothing
end

end # module HydrodynamicsModule