# src/Hydrodynamics.jl

module HydrodynamicsModule

export update_hydrodynamics!, update_hydrodynamics_placeholder!

using ..HydrodynamicTransport.ModelStructs
using NCDatasets
using Dates

function find_time_index(time_dim_seconds::Vector{<:Real}, current_time_seconds::Real)
    _, index = findmin(abs.(time_dim_seconds .- current_time_seconds))
    return index
end

# The placeholder functions for CartesianGrid are already ghost-cell aware and remain unchanged.
function update_hydrodynamics_placeholder!(state::State, grid::CartesianGrid, time::Float64)
    ng = grid.ng
    nx, ny, nz = grid.dims
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    dx = (grid.x[ng+2, ng+1, 1] - grid.x[ng+1, ng+1, 1])
    dy = (grid.y[ng+1, ng+2, 1] - grid.y[ng+1, ng+1, 1])
    Lx = nx * dx
    Ly = ny * dy
    center_x = Lx / 2
    center_y = Ly / 2

    period = 200.0
    omega = 2π / period
    
    # Calculate u on u-faces
    for k in 1:nz, j_glob in 1:ny_tot, i_glob in 1:nx_tot+1
        i_phys_face = i_glob - ng - 0.5
        j_phys_center = j_glob - ng
        x_coord_face = i_phys_face * dx
        y_coord_center = (j_phys_center - 0.5) * dy
        rx = x_coord_face - center_x; ry = y_coord_center - center_y
        state.u[i_glob, j_glob, k] = -omega * ry
    end

    # Calculate v on v-faces
    for k in 1:nz, j_glob in 1:ny_tot+1, i_glob in 1:nx_tot
        i_phys_center = i_glob - ng
        j_phys_face = j_glob - ng - 0.5
        x_coord_center = (i_phys_center - 0.5) * dx
        y_coord_face = j_phys_face * dy
        rx = x_coord_center - center_x; ry = y_coord_face - center_y
        state.v[i_glob, j_glob, k] = omega * rx
    end
    state.w .= 0.0
end

function update_hydrodynamics_placeholder!(state::State, grid::CurvilinearGrid, time::Float64)
    @warn "Placeholder hydrodynamics for CurvilinearGrid with ghost cells is not yet implemented. Setting velocities to zero."
    state.u .= 0.0; state.v .= 0.0; state.w .= 0.0
end


# --- Refactored Real Data Hydrodynamics ---
function update_hydrodynamics!(state::State, grid::CurvilinearGrid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64)
    ng = grid.ng
    
    # --- 1. Find the correct time index in the NetCDF file ---
    time_var_name = get(hydro_data.var_map, :time, "time")
    time_dim_raw = ds[time_var_name][:]
    
    if eltype(time_dim_raw) <: DateTime
        t0 = time_dim_raw[1]
        time_dim_seconds = [(dt - t0).value / 1000.0 for dt in time_dim_raw]
    else
        time_dim_seconds = time_dim_raw
    end
    
    time_idx = find_time_index(time_dim_seconds, time)
    
    # --- 2. Define which state fields to load from which NetCDF variables ---
    fields_to_load = [
        (state.u, :u, "xi_u", "eta_u", "s_rho"),
        (state.v, :v, "xi_v", "eta_v", "s_rho"),
        (state.w, :w, "xi_rho", "eta_rho", "s_w"),
        (state.temperature, :temp, "xi_rho", "eta_rho", "s_rho"),
        (state.salinity, :salt, "xi_rho", "eta_rho", "s_rho"),
    ]

    # --- 3. Loop through fields, create views, and read data into them ---
    for (state_field, standard_name, x_dim, y_dim, z_dim) in fields_to_load
        if haskey(hydro_data.var_map, standard_name)
            nc_var_name = hydro_data.var_map[standard_name]
            if haskey(ds, nc_var_name)
                # Get physical dimensions from the file
                nx_phys, ny_phys = ds.dim[x_dim], ds.dim[y_dim]
                nz_phys = ds.dim[z_dim]
                
                # Create a view into the physical interior of the state array
                # Note: Assumes u/v points align with rho points for ghost cells, which is standard
                interior_view = view(state_field, ng+1:nx_phys+ng, ng+1:ny_phys+ng, 1:nz_phys)

                # Read data slice from NetCDF and load it into the view
                data_slice = ds[nc_var_name][:, :, :, time_idx]
                interior_view .= coalesce.(data_slice, 0.0)
            else
                @warn "Variable '$(nc_var_name)' not found in NetCDF file. Skipping."
            end
        end
    end
    return nothing
end

end # module HydrodynamicsModule

