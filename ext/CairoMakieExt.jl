module CairoMakieExt

# --- Import the necessary packages ---
# The parent package, HydrodynamicTransport, is available automatically.
using CairoMakie
using HydrodynamicTransport
using Printf

# We need to explicitly import the functions we are adding methods to.
import HydrodynamicTransport: plot_state, plot_grid
# Also import types from the parent package.
using HydrodynamicTransport.ModelStructs


"""
    plot_state(grid::CurvilinearGrid, state::State, tracer_name::Symbol; 
               output_path="final_state.png", 
               show_velocity=true)

Creates a high-quality 2D plot of a tracer's concentration and the velocity field
for a given simulation state, saving it to a file. This function is only available
when `CairoMakie` is loaded.
"""
function HydrodynamicTransport.plot_state(grid::CurvilinearGrid, state::State, tracer_name::Symbol; 
                                          output_path="final_state.png", 
                                          show_velocity=true)

    @info "Generating Makie plot for state at t=$(state.time)s..."
    
    # Extract coordinate and tracer data for the surface layer
    lon = grid.lon_rho; lat = grid.lat_rho
    tracer_data = state.tracers[tracer_name][:,:,grid.nz]

    fig = Figure(size = (1000, 800))
    ax = Axis(fig[1, 1],
        title = "State at t = $(round(state.time/3600, digits=1)) hours",
        xlabel = "Longitude / X", ylabel = "Latitude / Y", aspect = DataAspect()
    )

    # Use contourf for filled concentration bands
    max_val = max(maximum(tracer_data), 1e-9)
    cf = contourf!(ax, lon, lat, tracer_data, colormap=:viridis, levels=20, colorrange=(0, max_val))
    Colorbar(fig[1, 2], cf, label="$tracer_name Concentration")

    if show_velocity
        # Rotate and plot velocity arrows
        u_east, v_north = rotate_velocities_to_geographic(grid, state.u, state.v)
        skip = floor(Int, max(grid.nx, grid.ny) / 20) # Keep arrow density reasonable
        arrows!(ax,
            lon[1:skip:end, 1:skip:end], lat[1:skip:end, 1:skip:end],
            u_east[1:skip:end, 1:skip:end, grid.nz], v_north[1:skip:end, 1:skip:end, grid.nz],
            lengthscale = 0.1, arrowsize = 7, color = :white, linewidth=0.8
        )
    end
    
    try
        save(output_path, fig)
        println("✅ Plot saved successfully to '$output_path'")
    catch e
        println("⚠️  Could not save Makie plot. Error: $e")
    end
end


"""
    plot_grid(grid::CurvilinearGrid; output_path="grid_layout.png")

Creates a high-quality 2D plot of a curvilinear grid's bathymetry and land mask,
saving it to a file. This function is only available when `CairoMakie` is loaded.
"""
function HydrodynamicTransport.plot_grid(grid::CurvilinearGrid; output_path="grid_layout.png")
    @info "Generating realistic grid visualization with Makie..."
    
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
        title = "Grid Layout and Bathymetry",
        xlabel = "Longitude / X", ylabel = "Latitude / Y", aspect = DataAspect()
    )

    # Plot bathymetry where there is water
    h_wet = ifelse.(grid.mask_rho, grid.h, NaN)
    cf = heatmap!(ax, grid.lon_rho, grid.lat_rho, h_wet, colormap=:deep)
    Colorbar(fig[1, 2], cf, label="Depth (m)")
    
    try
        save(output_path, fig)
        println("✅ Grid plot saved successfully to '$output_path'")
    catch e
        println("⚠️  Could not save Makie plot. Error: $e")
    end
end


end # module HydrodynamicTransportCairoMakieExt
```

### Step 4: How to Use the New Functionality

Now, the user experience is seamless and intuitive:

**Scenario 1: User does not have `CairoMakie`**
```julia
using HydrodynamicTransport

# ... run simulation to get a `final_state` ...

# This line will now fail gracefully with a helpful message:
plot_state(grid, final_state, :Tracer)
# ERROR: Plotting functionality not loaded. To use `plot_state`, ...
