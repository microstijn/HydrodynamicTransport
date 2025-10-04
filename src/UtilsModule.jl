# src/UtilsModule.jl

module UtilsModule

export estimate_stable_timestep
export create_hydrodynamic_data_from_file
export lonlat_to_ij

using NCDatasets
using ..HydrodynamicTransport.ModelStructs

"""
    estimate_stable_timestep(hydro_data::HydrodynamicData; 
                             pm_var="pm", 
                             pn_var="pn", 
                             safety_factor=0.8,
                             time_samples=3)

Estimates a stable timestep (dt) based on the CFL condition. By default, it uses a
fast sampling method, checking a few time steps (`time_samples`) instead of the
entire dataset. To check the full dataset (slower but more accurate), set
`time_samples=nothing`.

# Arguments
- `hydro_data`: The `HydrodynamicData` object for the simulation.
- `pm_var`, `pn_var`: Names of the grid metric variables.
- `safety_factor`: Factor to reduce the calculated timestep for safety.
- `time_samples`: Number of time steps to sample. Set to `nothing` to scan the entire dataset.
"""
function estimate_stable_timestep(hydro_data::HydrodynamicData; 
                                 pm_var="pm", 
                                 pn_var="pn", 
                                 safety_factor=0.8,
                                 time_samples::Union{Int, Nothing}=3)
    
    filepath = hydro_data.filepath
    println("--- Estimating Stable Timestep from '$filepath' ---")
    
    u_var = get(hydro_data.var_map, :u, "u")
    v_var = get(hydro_data.var_map, :v, "v")

    local u_max, v_max
    
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
            # --- SLOW PATH: Iterate over the entire remote dataset (most accurate) ---
            println("Scanning entire dataset for maximum velocities (this may be slow)...")
            u_max = maximum(abs, ds[u_var]; init=0.0)
            v_max = maximum(abs, ds[v_var]; init=0.0)
        else
            # --- FAST PATH: Sample a few time steps (default) ---
            println("Sampling $time_samples time steps for maximum velocities...")
            time_dim = ds[u_var].dim[end]
            n_times = ds.dim[time_dim]
            
            indices_to_sample = round.(Int, range(1, stop=n_times, length=time_samples))
            
            u_max_samples = Float64[]
            v_max_samples = Float64[]
            
            # The number of dimensions can vary, find the time dimension index
            time_dim_idx = findfirst(d -> d == time_dim, ds[u_var].dim)

            for t_idx in unique(indices_to_sample)
                # Build the correct indexer for this variable's dimensions
                slicer = [(:) for _ in 1:length(ds[u_var].dim)]
                slicer[time_dim_idx] = t_idx
                
                # Load only this time slice into memory and find its max
                u_slice = ds[u_var][slicer...]
                v_slice = ds[v_var][slicer...]
                push!(u_max_samples, maximum(abs.(u_slice); init=0.0))
                push!(v_max_samples, maximum(abs.(v_slice); init=0.0))
            end
            
            u_max = maximum(u_max_samples)
            v_max = maximum(v_max_samples)
        end
        
        println("Maximum grid-aligned velocities found: u_max ≈ $(round(u_max, digits=2))m/s, v_max ≈ $(round(v_max, digits=2))m/s")
        close(ds)

        # 3. Apply the CFL condition
        if u_max < 1e-9 && v_max < 1e-9; @warn "Velocities are zero."; return Inf; end
        
        dt_cfl = 1 / (u_max / dx_min + v_max / dy_min)
        safe_dt = dt_cfl * safety_factor
        
        println("--------------------------------------------------")
        println("Recommended stable timestep (dt): $(round(safe_dt, digits=2)) seconds")
        println(" (Based on a CFL safety factor of $safety_factor)")
        println("--------------------------------------------------")
        
        return safe_dt

    catch e
        println("Error during timestep estimation: $e"); return -1.0
    end
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

    # Define the search criteria for each target variable
    # The search is ordered by priority: standard_name, long_name, then variable name.
    search_patterns = Dict(
        :u => [("standard_name", "sea_water_x_velocity"), ("long_name", "u-velocity"), ("var_name", ("u", "U"))],
        :v => [("standard_name", "sea_water_y_velocity"), ("long_name", "v-velocity"), ("var_name", ("v", "V"))],
        :salt => [("standard_name", "sea_water_salinity"), ("long_name", "salinity"), ("var_name", ("salt", "sal", "SAL"))],
        :temp => [("standard_name", "sea_water_temperature"), ("long_name", "temperature"), ("var_name", ("temp", "TEMP"))],
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
                        variable_map[target_symbol] = var_name
                        println("  ✓ Found :$(target_symbol) -> '$(var_name)' (matched by variable name)")
                        found = true
                        break
                    end
                else # Search attributes
                    if haskey(ds[var_name].attrib, search_type)
                        attr_value = lowercase(ds[var_name].attrib[search_type])
                        if occursin(pattern, attr_value)
                            variable_map[target_symbol] = var_name
                            println("  ✓ Found :$(target_symbol) -> '$(var_name)' (matched by attribute '$(search_type)')")
                            found = true
                            break
                        end
                    end
                end
            end
        end
        if !found
            println("  - Warning: Could not find a variable for :$(target_symbol)")
        end
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
    
    # Determine the geographic bounding box of the physical grid
    physical_lon = view(grid.lon_rho, ng+1:nx+ng, ng+1:ny+ng)
    physical_lat = view(grid.lat_rho, ng+1:nx+ng, ng+1:ny+ng)
    
    lon_min, lon_max = extrema(physical_lon)
    lat_min, lat_max = extrema(physical_lat)

    # Check if the target point is within the bounding box
    if !(lon_min <= lon <= lon_max && lat_min <= lat <= lat_max)
        @warn "Target coordinates ($lon, $lat) are outside the grid's geographic bounding box."
        return nothing
    end

    min_dist_sq = Inf
    best_i, best_j = -1, -1
    is_tie = false

    for j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        # Only consider water points in the search
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
        @warn "Target coordinates ($lon, $lat) are within the bounding box, but no water cells were found nearby. The closest cells may all be land."
        return nothing
    end
    
    if is_tie
        @warn "Multiple grid points are equidistant to the target coordinates ($lon, $lat). " *
              "Returning the first match found: (i=$best_i, j=$best_j)."
    end

    return best_i, best_j
end

end # module UtilsModule