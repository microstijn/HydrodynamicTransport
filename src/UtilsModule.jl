# src/UtilsModule.jl

module UtilsModule

export estimate_stable_timestep
export create_hydrodynamic_data_from_file
export lonlat_to_ij
export calculate_max_cfl_term

using Base.Threads
using NCDatasets
using Dates
using ..HydrodynamicTransport.ModelStructs

"""
    calculate_max_cfl_term(state::State, grid::CurvilinearGrid)

Scans the grid to find the maximum value of (|u|/dx + |v|/dy), which is the
inverse of the Courant-limited timestep. This is used by the adaptive
time-stepping algorithm to validate a completed step.

This function is multithreaded for performance.

# Arguments
- `state::State`: The current model state, containing the velocity fields.
- `grid::CurvilinearGrid`: The model grid, containing grid metrics.

# Returns
- `Float64`: The maximum CFL term found across all water cells in the grid.
"""
function calculate_max_cfl_term(state::State, grid::CurvilinearGrid)
    u, v, w, pm, pn = state.u, state.v, state.w, grid.pm, grid.pn
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
    
    max_cfl_term = Threads.Atomic{Float64}(0.0)

    Threads.@threads for j in 1:ny
        for i in 1:nx
            i_glob, j_glob = i + ng, j + ng
            
            if grid.mask_rho[i_glob, j_glob]
                max_cfl_in_column = 0.0
                for k in 1:nz  # <-- THIS IS THE CRITICAL FIX
                    # 1. Horizontal CFL for this cell (i,j,k)
                    u_center = 0.5 * (u[i_glob, j_glob, k] + u[i_glob+1, j_glob, k])
                    v_center = 0.5 * (v[i_glob, j_glob, k] + v[i_glob, j_glob+1, k])
                    cfl_term_horiz = abs(u_center) * pm[i_glob, j_glob] + abs(v_center) * pn[i_glob, j_glob]

                    # 2. Vertical CFL for this cell (i,j,k)
                    dz = abs(grid.z_w[k+1] - grid.z_w[k])
                    # Use max face velocity (more conservative)
                    w_face_vel = max(abs(w[i_glob, j_glob, k]), abs(w[i_glob, j_glob, k+1]))
                    cfl_term_vert = w_face_vel / dz

                    # 3. Total 3D CFL for this cell
                    total_cfl_term = cfl_term_horiz + cfl_term_vert
                    
                    if total_cfl_term > max_cfl_in_column
                        max_cfl_in_column = total_cfl_term
                    end
                end
                Threads.atomic_max!(max_cfl_term, max_cfl_in_column)
            end
        end
    end
    return max_cfl_term[]
end

# Add the corresponding 3D method for CartesianGrid
function calculate_max_cfl_term(state::State, grid::CartesianGrid)
    u, v, w = state.u, state.v, state.w
    ng, (nx, ny, nz) = grid.ng, grid.dims
    
    # Calculate dx, dy, dz directly and correctly
    dx = (grid.x[ng+2, ng+1, 1] - grid.x[ng+1, ng+1, 1])
    dy = (grid.y[ng+1, ng+2, 1] - grid.y[ng+1, ng+1, 1])
    
    max_cfl_term = Threads.Atomic{Float64}(0.0)

    Threads.@threads for j in 1:ny
        for i in 1:nx
            i_glob, j_glob = i + ng, j + ng

            if grid.mask[i_glob, j_glob, 1] 
                max_cfl_in_column = 0.0
                for k in 1:nz 
                    # 1. Horizontal CFL
                    u_center = 0.5 * (u[i_glob, j_glob, k] + u[i_glob+1, j_glob, k])
                    v_center = 0.5 * (v[i_glob, j_glob, k] + v[i_glob, j_glob+1, k])
                    cfl_term_horiz = abs(u_center) / dx + abs(v_center) / dy

                    # 2. Vertical CFL
                    # --- THIS IS THE FIX ---
                    # Calculate dz robustly using volume and face area (using top face k+1)
                    dz = grid.volume[i_glob,j_glob,k] / grid.face_area_z[i_glob,j_glob,k+1]
                    # --- END FIX ---
                    
                    w_face_vel = max(abs(w[i_glob, j_glob, k]), abs(w[i_glob, j_glob, k+1]))
                    cfl_term_vert = w_face_vel / dz

                    # 3. Total 3D CFL
                    total_cfl_term = cfl_term_horiz + cfl_term_vert

                    if total_cfl_term > max_cfl_in_column
                        max_cfl_in_column = total_cfl_term
                    end
                end
                Threads.atomic_max!(max_cfl_term, max_cfl_in_column)
            end
        end
    end
    return max_cfl_term[]
end

"""
    estimate_stable_timestep(hydro_data; advection_scheme, start_time, end_time, ...)

Estimates a recommended timestep (dt) based on the chosen advection scheme and time window.

By default, it uses a fast sampling method to find maximum velocities within the specified
time window. To check the full time window (slower but more accurate), set `time_samples=nothing`.

# Arguments
- `hydro_data`: The `HydrodynamicData` object for the simulation.
- `advection_scheme::Symbol`: The advection scheme to be used (`:TVD`, `:UP3`, `:ImplicitADI`).
- `start_time::Union{Float64, Nothing}`: The simulation start time in seconds to begin sampling. If `nothing`, starts from the beginning of the file.
- `end_time::Union{Float64, Nothing}`: The simulation end time in seconds to stop sampling. If `nothing`, samples until the end of the file.
- `dx_var::String`, `dy_var::String`: Names of the grid cell size variables in the NetCDF file.
- `safety_factor`: For explicit schemes, the factor to reduce the calculated CFL timestep for safety (e.g., 0.8).
- `CFL_acc`: For implicit schemes, the desired "accuracy Courant number" to base the recommendation on (e.g., 5.0).
- `time_samples`: Number of time steps to sample for velocity checks. `nothing` scans the entire specified time window.
"""
function estimate_stable_timestep(hydro_data::HydrodynamicData; 
                                 advection_scheme::Symbol=:TVD,
                                 start_time::Union{Float64, Nothing}=nothing,
                                 end_time::Union{Float64, Nothing}=nothing,
                                 dx_var::String="dx", 
                                 dy_var::String="dy", 
                                 safety_factor=0.8,
                                 CFL_acc::Float64=5.0,
                                 time_samples::Union{Int, Nothing}=3)
    
    filepath = hydro_data.filepath
    println("--- Estimating Timestep from '$filepath' (Scheme: $advection_scheme) ---")
    
    u_var = get(hydro_data.var_map, :u, "u")
    v_var = get(hydro_data.var_map, :v, "v")
    time_var = get(hydro_data.var_map, :time, "ocean_time")

    local u_max, v_max, dx_min, dy_min
    
    try
        ds = NCDataset(filepath)
        # 1. Find minimum grid spacing
        if !haskey(ds, dx_var) || !haskey(ds, dy_var); error("Grid spacing variables '$dx_var' or '$dy_var' not found."); end
        dx_min = minimum(filter(x -> !ismissing(x) && x > 0, ds[dx_var][:]))
        dy_min = minimum(filter(x -> !ismissing(x) && x > 0, ds[dy_var][:]))
        println("Minimum grid spacing: dx ≈ $(round(dx_min, digits=2))m, dy ≈ $(round(dy_min, digits=2))m")

        # 2. Find maximum velocities
        if !haskey(ds, u_var) || !haskey(ds, v_var); error("Velocity variables '$u_var' or '$v_var' not found."); end
        
        # Convert DateTime objects to elapsed seconds for comparison
        time_dim_raw = ds[time_var][:]
        time_dim_seconds = if eltype(time_dim_raw) <: Dates.AbstractTime
            t0 = time_dim_raw[1]
            [(dt - t0).value / 1000.0 for dt in time_dim_raw]
        else
            time_dim_raw
        end

        # Find the start and end indices corresponding to the provided times
        start_idx = start_time !== nothing ? searchsortedfirst(time_dim_seconds, start_time) : 1
        end_idx = end_time !== nothing ? searchsortedlast(time_dim_seconds, end_time) : length(time_dim_seconds)
        if end_idx < start_idx; end_idx = start_idx; end

        # Robustness check: if the window is too small, expand it to get a good sample.
        num_available_steps = end_idx - start_idx + 1
        num_samples = time_samples === nothing ? num_available_steps : time_samples
        
        if num_available_steps < num_samples
            println("Warning: Specified time window is too small or data resolution is too coarse.")
            new_end_idx = min(start_idx + num_samples - 1, length(time_dim_seconds))
            println("Expanding search window from indices [$start_idx, $end_idx] to [$start_idx, $new_end_idx].")
            end_idx = new_end_idx
        end

        println("Sampling for velocities between time index $start_idx and $end_idx...")
        
        if time_samples === nothing
            slicer = (fill(Colon(), ndims(ds[u_var]) - 1)..., start_idx:end_idx)
            u_max = maximum(abs, skipmissing(ds[u_var][slicer...]));
            v_max = maximum(abs, skipmissing(ds[v_var][slicer...]));
        else
            indices_to_sample = round.(Int, range(start_idx, stop=end_idx, length=num_samples))
            u_max_samples, v_max_samples = Float64[], Float64[]
            for t_idx in unique(indices_to_sample)
                slicer = (fill(Colon(), ndims(ds[u_var]) - 1)..., t_idx)
                push!(u_max_samples, maximum(abs, skipmissing(ds[u_var][slicer...])))
                push!(v_max_samples, maximum(abs, skipmissing(ds[v_var][slicer...])))
            end
            u_max = maximum(u_max_samples); v_max = maximum(v_max_samples)
        end
        println("Maximum grid-aligned velocities found in window: u_max ≈ $(round(u_max, digits=2))m/s, v_max ≈ $(round(v_max, digits=2))m/s")
        close(ds)
    catch e
        println("Error during data loading for timestep estimation: $e"); return -1.0
    end

    # 3. Calculate timestep
    if u_max < 1e-9 && v_max < 1e-9
        @warn "Velocities are zero or negligible; cannot estimate timestep."
        return Inf
    end

    cfl_denominator = (u_max / dx_min + v_max / dy_min)
    if advection_scheme in (:TVD, :UP3)
        recommended_dt = (1 / cfl_denominator) * safety_factor
        println("--------------------------------------------------")
        println("Recommended STABLE timestep (dt): $(round(recommended_dt, digits=2)) seconds")
        println(" (Based on CFL stability limit with safety factor $safety_factor)")
        println("--------------------------------------------------")
    elseif advection_scheme == :ImplicitADI
        recommended_dt = CFL_acc / cfl_denominator
        println("--------------------------------------------------")
        println("Recommended ACCURACY timestep (dt): $(round(recommended_dt, digits=2)) seconds")
        println(" (Based on accuracy Courant number CFL_acc = $CFL_acc)")
        println("--------------------------------------------------")
    else
        error("Unknown advection scheme '$advection_scheme' for timestep estimation.")
    end
    
    return recommended_dt
end


"""
    create_hydrodynamic_data_from_file(filepath::String) -> HydrodynamicData

Automatically inspects a NetCDF file and attempts to generate the `variable_map`
required by `HydrodynamicData`.

It searches for common variable names and attributes (like `standard_name` or `long_name`)
to identify velocities, salinity, temperature, and time.

# Arguments
- `filepath`: Path to the NetCDF grid/history file.

# Returns
- A configured `HydrodynamicData` object.
"""
function create_hydrodynamic_data_from_file(filepath::String)
    println("--- Autodetecting variables from '$filepath' ---")
    variable_map = Dict{Symbol, String}()
    
    # --- FIX: Prioritize 3D baroclinic velocities (uz, vz) over 2D barotropic (u, v) ---
    search_patterns = Dict(
        :u => [("standard_name", "sea_water_x_velocity"), ("long_name", "u-velocity"), ("var_name", ("uz", "u", "U"))],
        :v => [("standard_name", "sea_water_y_velocity"), ("long_name", "v-velocity"), ("var_name", ("vz", "v", "V"))],
        :salt => [("standard_name", "sea_water_salinity"), ("long_name", "salinity"), ("var_name", ("salt", "sal", "SAL"))],
        :temp => [("standard_name", "sea_water_potential_temperature"), ("long_name", "temperature"), ("var_name", ("temp", "TEMP"))],
        :time => [("standard_name", "time"), ("long_name", "time"), ("var_name", ("time", "ocean_time"))]
    )
    
    ds = NCDataset(filepath)
    file_vars = keys(ds)
    for (target_symbol, patterns) in search_patterns
        found = false
        for (search_type, pattern) in patterns
            if found; break; end
            for var_name in file_vars
                if search_type == "var_name"
                    if lowercase(var_name) in pattern
                        variable_map[target_symbol] = var_name; println("  ✓ Found :$(target_symbol) -> '$(var_name)' (matched by variable name)"); found = true; break
                    end
                else 
                    if haskey(ds[var_name].attrib, search_type)
                        attr_value = lowercase(ds[var_name].attrib[search_type])
                        # Add a check to exclude barotropic velocities when searching for standard_name
                        if occursin(pattern, attr_value) && !occursin("barotropic", attr_value)
                            variable_map[target_symbol] = var_name; println("  ✓ Found :$(target_symbol) -> '$(var_name)' (matched by attribute '$(search_type)')"); found = true; break
                        end
                    end
                end
            end
        end
        if !found; println("  - Warning: Could not find a variable for :$(target_symbol)"); end
    end
    close(ds)
    return HydrodynamicData(filepath, variable_map)
end

"""
    lonlat_to_ij(grid::CurvilinearGrid, lon::Float64, lat::Float64) -> Union{Tuple{Int, Int}, Nothing}

Finds the physical grid indices (i, j) of the water cell center closest to the target
geographic coordinates (longitude, latitude).

If the target coordinates are outside the grid's geographic bounding box, or if no
water cells are found nearby, it returns `nothing` and issues a warning.
"""
function lonlat_to_ij(grid::CurvilinearGrid, lon::Float64, lat::Float64)
    nx, ny = grid.nx, grid.ny
    ng = grid.ng

    # --- FIX: Calculate the bounding box using ONLY the valid water points ---
    # This prevents missing/fill values from skewing the calculation.
    water_points_mask = view(grid.mask_rho, ng+1:nx+ng, ng+1:ny+ng)
    if !any(water_points_mask)
        @warn "The provided grid has no water points (all cells are masked as land)."
        return nothing
    end

    lon_phys = view(grid.lon_rho, ng+1:nx+ng, ng+1:ny+ng)
    lat_phys = view(grid.lat_rho, ng+1:nx+ng, ng+1:ny+ng)
    
    lon_min, lon_max = extrema(lon_phys[water_points_mask])
    lat_min, lat_max = extrema(lat_phys[water_points_mask])

    if !(lon_min <= lon <= lon_max) || !(lat_min <= lat <= lat_max)
        @warn "Target coordinates ($lon, $lat) are outside the grid's geographic bounding box of water points. Lon: [$lon_min, $lon_max], Lat: [$lat_min, $lat_max]."
        return nothing
    end
    
    min_dist_sq = Inf
    best_i, best_j = -1, -1
    is_tie = false

    @inbounds for j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        # Search only over water cells
        if grid.mask_rho[i_glob, j_glob]
            dist_sq = (grid.lon_rho[i_glob, j_glob] - lon)^2 + (grid.lat_rho[i_glob, j_glob] - lat)^2
            
            if dist_sq < min_dist_sq
                min_dist_sq = dist_sq
                best_i, best_j = i_phys, j_phys
                is_tie = false
            elseif dist_sq == min_dist_sq
                is_tie = true
            end
        end
    end
    
    if best_i == -1
        @warn "Target coordinates ($lon, $lat) are within the bounding box but no water cells were found."
        return nothing
    end

    if is_tie
        @warn "Multiple grid points are equidistant to the target coordinates ($lon, $lat). " *
              "Returning the first match found: (i=$best_i, j=$best_j)."
    end

    return best_i, best_j
end


end # module UtilsModule