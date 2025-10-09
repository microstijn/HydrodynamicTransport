# src/UtilsModule.jl

module UtilsModule

export estimate_stable_timestep
export create_hydrodynamic_data_from_file
export lonlat_to_ij

using NCDatasets
using ..HydrodynamicTransport.ModelStructs

"""
    estimate_stable_timestep(hydro_data::HydrodynamicData; ...)

Estimates a recommended timestep (dt) based on the chosen advection scheme.

- For **explicit schemes** (`:TVD`, `:UP3`), it calculates a stability-limited
  timestep based on the Courant-Friedrichs-Lewy (CFL) condition.
- For **implicit schemes** (`:ImplicitADI`), it provides an accuracy-limited
  timestep recommendation, as the scheme is unconditionally stable.

By default, it uses a fast sampling method to find maximum velocities. To check
the full dataset (slower but more accurate), set `time_samples=nothing`.

# Arguments
- `hydro_data`: The `HydrodynamicData` object for the simulation.
- `advection_scheme::Symbol`: The advection scheme to be used (`:TVD`, `:UP3`, `:ImplicitADI`). Defaults to `:TVD`.
- `pm_var`, `pn_var`: Names of the grid metric variables in the NetCDF file.
- `safety_factor`: For explicit schemes, the factor to reduce the calculated CFL timestep for safety (e.g., 0.8).
- `CFL_acc`: For implicit schemes, the desired "accuracy Courant number" to base the recommendation on (e.g., 5.0).
- `time_samples`: Number of time steps to sample for velocity checks. `nothing` scans the entire dataset.
"""
function estimate_stable_timestep(hydro_data::HydrodynamicData; 
                                 advection_scheme::Symbol=:TVD,
                                 pm_var="pm", 
                                 pn_var="pn", 
                                 safety_factor=0.8,
                                 CFL_acc::Float64=5.0,
                                 time_samples::Union{Int, Nothing}=3)
    
    filepath = hydro_data.filepath
    println("--- Estimating Timestep from '$filepath' (Scheme: $advection_scheme) ---")
    
    u_var = get(hydro_data.var_map, :u, "u")
    v_var = get(hydro_data.var_map, :v, "v")

    local u_max, v_max, dx_min, dy_min
    
    try
        ds = NCDataset(filepath)
        # 1. Find minimum grid spacing (fast, as pm/pn are 2D)
        if !haskey(ds, pm_var) || !haskey(ds, pn_var); error("Grid metric variables '$pm_var' or '$pn_var' not found."); end
        dx_min = 1 / maximum(ds[pm_var]; init=0.0)
        dy_min = 1 / maximum(ds[pn_var]; init=0.0)
        println("Minimum grid spacing: dx ≈ $(round(dx_min, digits=2))m, dy ≈ $(round(dy_min, digits=2))m")

        # 2. Find maximum velocities
        if !haskey(ds, u_var) || !haskey(ds, v_var); error("Velocity variables '$u_var' or '$v_var' not found."); end
        if time_samples === nothing
            println("Scanning entire dataset for maximum velocities (this may be slow)...")
            u_max = maximum(abs, ds[u_var]; init=0.0)
            v_max = maximum(abs, ds[v_var]; init=0.0)
        else
            println("Sampling $time_samples time steps for maximum velocities...")
            time_dim = ds[u_var].dim[end]
            n_times = ds.dim[time_dim]
            indices_to_sample = round.(Int, range(1, stop=n_times, length=time_samples))
            u_max_samples = Float64[]
            v_max_samples = Float64[]
            time_dim_idx = findfirst(d -> d == time_dim, ds[u_var].dim)
            for t_idx in unique(indices_to_sample)
                slicer = [(:) for _ in 1:length(ds[u_var].dim)]
                slicer[time_dim_idx] = t_idx
                u_slice = ds[u_var][slicer...]; v_slice = ds[v_var][slicer...]
                push!(u_max_samples, maximum(abs.(u_slice); init=0.0))
                push!(v_max_samples, maximum(abs.(v_slice); init=0.0))
            end
            u_max = maximum(u_max_samples); v_max = maximum(v_max_samples)
        end
        println("Maximum grid-aligned velocities found: u_max ≈ $(round(u_max, digits=2))m/s, v_max ≈ $(round(v_max, digits=2))m/s")
        close(ds)
    catch e
        println("Error during data loading for timestep estimation: $e"); return -1.0
    end

    # 3. Calculate timestep based on the chosen scheme
    if u_max < 1e-9 && v_max < 1e-9
        @warn "Velocities are zero or negligible; cannot estimate timestep."
        return Inf
    end

    local recommended_dt::Float64
    cfl_denominator = (u_max / dx_min + v_max / dy_min)

    if advection_scheme in (:TVD, :UP3)
        dt_cfl = 1 / cfl_denominator
        recommended_dt = dt_cfl * safety_factor
        println("--------------------------------------------------")
        println("Recommended STABLE timestep (dt): $(round(recommended_dt, digits=2)) seconds")
        println(" (Based on CFL stability limit with safety factor $safety_factor)")
        println("--------------------------------------------------")
    elseif advection_scheme == :ImplicitADI
        recommended_dt = CFL_acc / cfl_denominator
        println("--------------------------------------------------")
        println("Recommended ACCURACY timestep (dt): $(round(recommended_dt, digits=2)) seconds")
        println(" (Based on accuracy Courant number CFL_acc = $CFL_acc)")
        println(" (The :ImplicitADI scheme is unconditionally stable)")
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