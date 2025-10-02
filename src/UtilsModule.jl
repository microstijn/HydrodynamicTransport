# src/UtilsModule.jl

module UtilsModule

export estimate_stable_timestep

using NCDatasets

"""
    estimate_stable_timestep(filepath::String; 
                             u_var="u", 
                             v_var="v", 
                             pm_var="pm", 
                             pn_var="pn", 
                             safety_factor=0.8)

Estimates a stable timestep (dt) for the advection scheme based on the 
Courant-Friedrichs-Lewy (CFL) condition.

It reads the maximum velocities and minimum grid spacings from a NetCDF file.

# Arguments
- `filepath`: Path to the NetCDF grid/history file.
- `u_var`, `v_var`, `pm_var`, `pn_var`: Names of the u/v velocity and grid metric variables.
- `safety_factor`: A factor (e.g., 0.8) to reduce the calculated timestep for an extra margin of safety.

# Returns
- A `Float64` representing the recommended stable timestep in seconds.
"""
function estimate_stable_timestep(filepath::String; 
                                 u_var="u", 
                                 v_var="v", 
                                 pm_var="pm", 
                                 pn_var="pn", 
                                 safety_factor=0.8)
    
    println("--- Estimating Stable Timestep from '$filepath' ---")
    
    local u_max, v_max, dx_min, dy_min
    
    try
        ds = NCDataset(filepath)

        # 1. Find the minimum grid spacing from the grid metrics (pm, pn)
        # dx = 1/pm, dy = 1/pn
        if !haskey(ds, pm_var) || !haskey(ds, pn_var)
             error("Grid metric variables '$pm_var' or '$pn_var' not found in file.")
        end
        dx_min = 1 / maximum(ds[pm_var][:])
        dy_min = 1 / maximum(ds[pn_var][:])
        
        println("Minimum grid spacing: dx ≈ $(round(dx_min, digits=2))m, dy ≈ $(round(dy_min, digits=2))m")

        # 2. Find the maximum grid-aligned velocities
        if !haskey(ds, u_var) || !haskey(ds, v_var)
            error("Velocity variables '$u_var' or '$v_var' not found in file.")
        end
        u_max = maximum(abs.(ds[u_var][:]))
        v_max = maximum(abs.(ds[v_var][:]))

        println("Maximum grid-aligned velocities: u_max ≈ $(round(u_max, digits=2))m/s, v_max ≈ $(round(v_max, digits=2))m/s")
        
        close(ds)

        # 3. Apply the CFL condition
        # A conservative approach for stability in 2D
        if u_max < 1e-9 && v_max < 1e-9
            @warn "Velocities are zero. Timestep calculation is not meaningful."
            return Inf
        end
        
        dt_cfl = 1 / (u_max / dx_min + v_max / dy_min)
        safe_dt = dt_cfl * safety_factor
        
        println("--------------------------------------------------")
        println("Recommended stable timestep (dt): $(round(safe_dt, digits=2)) seconds")
        println(" (Based on a CFL safety factor of $safety_factor)")
        println("--------------------------------------------------")
        
        return safe_dt

    catch e
        println("Error during timestep estimation: $e")
        return -1.0 # Return an indicative error value
    end
end

end # module UtilsModule
