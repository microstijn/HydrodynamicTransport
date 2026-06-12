# src/Hydrodynamics.jl

module HydrodynamicsModule

export update_hydrodynamics!, update_hydrodynamics_placeholder!

using ..HydrodynamicTransport.ModelStructs
using NCDatasets
using Dates

# --- The placeholder functions for CartesianGrid remain unchanged ---
function update_hydrodynamics_placeholder!(state::State, grid::CartesianGrid, time::Float64)
    ng = grid.ng
    nx, ny, nz = grid.dims
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    dx = (grid.x[ng+2, ng+1, 1] - grid.x[ng+1, ng+1, 1])
    dy = (grid.y[ng+1, ng+2, 1] - grid.y[ng+1, ng+1, 1])
    Lx = nx * dx; Ly = ny * dy
    center_x = Lx / 2; center_y = Ly / 2
    period = 200.0; omega = 2π / period
    
    for k in 1:nz, j_glob in 1:ny_tot, i_glob in 1:nx_tot+1
        i_phys_face = i_glob - ng - 0.5; j_phys_center = j_glob - ng
        x_coord_face = i_phys_face * dx; y_coord_center = (j_phys_center - 0.5) * dy
        rx = x_coord_face - center_x; ry = y_coord_center - center_y
        state.u[i_glob, j_glob, k] = -omega * ry
    end

    for k in 1:nz, j_glob in 1:ny_tot+1, i_glob in 1:nx_tot
        i_phys_center = i_glob - ng; j_phys_face = j_glob - ng - 0.5
        x_coord_center = (i_phys_center - 0.5) * dx; y_coord_face = j_phys_face * dy
        rx = x_coord_center - center_x; ry = y_coord_face - center_y
        state.v[i_glob, j_glob, k] = omega * rx
    end
    state.w .= 0.0
    state.zeta .= 0.0
end

function update_hydrodynamics_placeholder!(state::State, grid::CurvilinearGrid, time::Float64)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    dx_approx = 1.0 / grid.pm[ng+nx÷2, ng+ny÷2]
    dy_approx = 1.0 / grid.pn[ng+nx÷2, ng+ny÷2]

    Lx = nx * dx_approx; Ly = ny * dy_approx
    center_x = Lx / 2; center_y = Ly / 2
    omega = 0.001

    # Set all velocities to zero initially
    state.u .= 0.0
    state.v .= 0.0

    # Calculate velocities for INTERIOR faces only
    for k in 1:nz
        # U-velocities (X-faces)
        for j_phys in 1:ny, i_phys in 2:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            y_center = (j_phys - 0.5) * dy_approx
            # FIX: Corrected sign for counter-clockwise rotation
            state.u[i_glob, j_glob, k] = omega * (y_center - center_y)
        end
        # V-velocities (Y-faces)
        for j_phys in 2:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            x_center = (i_phys - 0.5) * dx_approx
            state.v[i_glob, j_glob, k] = -omega * (x_center - center_x)
        end
    end
    
    state.w .= 0.0
    state.zeta .= 0.0
end


# Load (and coalesce) the bracketing time-slab for `idx` into the cache, but only if it
# isn't already cached. This is what removes the per-timestep NetCDF reads.
function _ensure_hydro_slab!(cache::HydroSlabCache, ds::NCDataset, hydro_data::HydrodynamicData, fields, idx::Int)
    haskey(cache.slabs, idx) && return
    d = Dict{Symbol, Array{Float64}}()
    for (_state_field, standard_name, has_z) in fields
        if haskey(hydro_data.var_map, standard_name)
            nc_var_name = hydro_data.var_map[standard_name]
            if haskey(ds, nc_var_name)
                d[standard_name] = has_z ? coalesce.(ds[nc_var_name][:, :, :, idx], 0.0) :
                                           coalesce.(ds[nc_var_name][:, :, idx], 0.0)
            end
        end
    end
    cache.slabs[idx] = d
    return
end

# --- Real-data hydrodynamics with temporal interpolation + cached time-slabs ---
# Numerically identical to the previous version; the only change is that the time axis is
# converted once and the bracketing slabs are read from disk only when their time index
# changes (then kept in hydro_data.cache), instead of being re-read every timestep.
function update_hydrodynamics!(state::State, grid::CurvilinearGrid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64)
    ng = grid.ng
    cache = hydro_data.cache

    # Convert the time axis once, then reuse from the cache.
    if cache.time_seconds === nothing
        time_var_name = get(hydro_data.var_map, :time, "time")
        time_dim_raw = ds[time_var_name][:]
        cache.time_seconds = if eltype(time_dim_raw) <: DateTime
            t0 = time_dim_raw[1]; [(dt - t0).value / 1000.0 for dt in time_dim_raw]
        else
            Float64.(time_dim_raw)
        end
    end
    time_dim_seconds = cache.time_seconds
    n_times = length(time_dim_seconds)

    local idx1, idx2, weight
    if time <= time_dim_seconds[1]; idx1 = 1; idx2 = 1; weight = 0.0
    elseif time >= time_dim_seconds[n_times]; idx1 = n_times; idx2 = n_times; weight = 0.0
    else; idx1 = searchsortedlast(time_dim_seconds, time); idx2 = idx1 + 1; t1 = time_dim_seconds[idx1]; t2 = time_dim_seconds[idx2]; time_interval = t2 - t1; weight = (time_interval > 1e-9) ? (time - t1) / time_interval : 0.0; end

    # (state_field, standard_name, has_z). zeta is a 2-D field broadcast across z layers.
    fields = ((state.u, :u, true), (state.v, :v, true), (state.temperature, :temp, true),
              (state.salinity, :salt, true), (state.zeta, :zeta, false))

    # Read bracketing slabs from disk only if not already cached, then bound cache size.
    _ensure_hydro_slab!(cache, ds, hydro_data, fields, idx1)
    idx2 != idx1 && _ensure_hydro_slab!(cache, ds, hydro_data, fields, idx2)
    if length(cache.slabs) > 4
        for k in collect(keys(cache.slabs)); (k == idx1 || k == idx2) || delete!(cache.slabs, k); end
    end

    slabs1 = cache.slabs[idx1]; slabs2 = cache.slabs[idx2]
    for (state_field, standard_name, has_z) in fields
        haskey(slabs1, standard_name) || continue
        data_slice1 = slabs1[standard_name]
        if has_z
            nx_phys, ny_phys, nz_phys = size(data_slice1)
            interior_view = view(state_field, ng+1:nx_phys+ng, ng+1:ny_phys+ng, 1:nz_phys)
            if weight > 1e-9
                interior_view .= (1.0 - weight) .* data_slice1 .+ weight .* slabs2[standard_name]
            else
                interior_view .= data_slice1
            end
        else
            nx_phys, ny_phys = size(data_slice1)
            interior_view_3d = view(state_field, ng+1:nx_phys+ng, ng+1:ny_phys+ng, :)
            interpolated_data_2d = (weight > 1e-9) ?
                ((1.0 - weight) .* data_slice1 .+ weight .* slabs2[standard_name]) : data_slice1
            for k in 1:size(interior_view_3d, 3)
                view(interior_view_3d, :, :, k) .= interpolated_data_2d
            end
        end
    end
    return nothing
end

end # module HydrodynamicsModule

