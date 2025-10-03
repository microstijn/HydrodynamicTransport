# src/UtilsModule.jl

module UtilsModule

export estimate_stable_timestep, create_hydrodynamic_data_from_file

using NCDatasets
using ..HydrodynamicTransport.ModelStructs

"""
    estimate_stable_timestep(filepath::String; ...)

Estimates a stable timestep (dt) for the advection scheme based on the 
Courant-Friedrichs-Lewy (CFL) condition.
... (full docstring from previous response) ...
"""
function estimate_stable_timestep(filepath::String; 
                                 u_var="u", v_var="v", pm_var="pm", pn_var="pn", 
                                 safety_factor=0.8)
    
    println("--- Estimating Stable Timestep from '$filepath' ---")
    local u_max, v_max, dx_min, dy_min
    try
        ds = NCDataset(filepath)
        if !haskey(ds, pm_var) || !haskey(ds, pn_var); error("Grid metric variables '$pm_var' or '$pn_var' not found."); end
        dx_min = 1 / maximum(ds[pm_var][:]); dy_min = 1 / maximum(ds[pn_var][:])
        println("Minimum grid spacing: dx ≈ $(round(dx_min, digits=2))m, dy ≈ $(round(dy_min, digits=2))m")
        if !haskey(ds, u_var) || !haskey(ds, v_var); error("Velocity variables '$u_var' or '$v_var' not found."); end
        u_max = maximum(abs.(ds[u_var][:]); init=0.0); v_max = maximum(abs.(ds[v_var][:]); init=0.0)
        println("Maximum grid-aligned velocities: u_max ≈ $(round(u_max, digits=2))m/s, v_max ≈ $(round(v_max, digits=2))m/s")
        close(ds)
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


end # module UtilsModule