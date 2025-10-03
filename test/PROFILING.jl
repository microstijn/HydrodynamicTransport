# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using Profile

# Import all necessary unexported functions for the simulation loop
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.HydrodynamicsModule

println("--- Performance Profiling Script for HydrodynamicTransport.jl ---")
println("This script will run a simulation and collect performance data.")
println("The output will show where the model spends the most time.\n")


# ==============================================================================
# --- 2. Define the Simulation Function to be Profiled ---
# ==============================================================================

# We wrap the main loop in a function. This is the target for our profiler.
function run_for_profiling(grid, state, sources, bcs, dt, end_time, advection_scheme)
    
    time_range = 0.0:dt:end_time
    
    for time in time_range
        if time == 0.0; continue; end
        state.time = time

        # Run one time step with the full sequence of operations
        apply_boundary_conditions!(state, grid, bcs)
        update_hydrodynamics_placeholder!(state, grid, time)
        horizontal_transport!(state, grid, dt, advection_scheme)
        vertical_transport!(state, grid, dt)
        source_sink_terms!(state, grid, sources, time, dt)
    end
    
    return state
end


# ==============================================================================
# --- 3. Setup a Realistic Test Case ---
# ==============================================================================

println("Setting up a realistic test case for profiling...")

# A larger grid gives more meaningful performance results
const nx, ny, nz = 150, 150, 10
const ng = 2

# Initialize grid and state
grid = initialize_cartesian_grid(nx, ny, nz, 1500.0, 1500.0, 100.0; ng=ng)
state = initialize_state(grid, (:C,))

# Define some sources and boundaries
sources = [PointSource(i=25, j=75, k=1, tracer_name=:C, influx_rate=(t)->100.0)]
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West)]

# Simulation parameters
const dt = 0.5
const end_time = 50.0 # Run for a short duration to keep profiling manageable


# ==============================================================================
# --- 4. Run and Analyze Profiles ---
# ==============================================================================

# --- Profile the :TVD Scheme ---
println("\n--- Profiling Simulation with :TVD Advection Scheme ---")
# First run is for JIT compilation to warm up the code
run_for_profiling(deepcopy(state), grid, sources, bcs, dt, 1.0, :TVD)

# Clear any previous profiling data and run the profiler
Profile.clear()
@profile run_for_profiling(deepcopy(state), grid, sources, bcs, dt, end_time, :TVD)

println("--- Results for :TVD Scheme ---")
# Print the text-based profile report
Profile.print(format=:flat, sortedby=:count, mincount=10)


# --- Profile the :UP3 Scheme ---
println("\n\n--- Profiling Simulation with :UP3 Advection Scheme ---")
# First run for JIT compilation
run_for_profiling(deepcopy(state), grid, sources, bcs, dt, 1.0, :UP3)

# Clear profiling data and run again
Profile.clear()
@profile run_for_profiling(deepcopy(state), grid, sources, bcs, dt, end_time, :UP3)

println("--- Results for :UP3 Scheme ---")
Profile.print(format=:flat, sortedby=:count, mincount=10)


# ==============================================================================
# --- 5. How to Interpret the Results ---
# ==============================================================================

println("""

--- How to Interpret the Profiling Results ---

1.  **The Output Format:** The list shows a "stack trace" of function calls. The number on the left is the number of "samples" collected while the program was executing code within that function and any functions it called. A higher number means more time was spent there.

2.  **Identifying Hotspots:** Look for the lines with the highest sample counts at the lowest indentation levels. These are your primary "hotspots." You should see `horizontal_transport!`, `vertical_transport!`, etc., near the top.

3.  **Comparing Schemes:** Compare the total samples for `run_for_profiling` between the TVD and UP3 runs. More importantly, look at the samples within `horizontal_transport!` and specifically within `advect_x_tvd!` vs. `advect_x_up3!`. This will give you a direct measure of their relative cost.

4.  **Memory Allocations (Indirect):** High sample counts inside garbage collection (`gc.c`) functions are a red flag for excessive memory allocation. Our previous optimizations should have minimized this, but it's always good to check.

5.  **(Optional) Visual Profiling:** For a much more intuitive view, run this script in an interactive Julia session and then use `ProfileView`.
    - Run the script.
    - After the TVD profile, type `using ProfileView; ProfileView.view()`
    - After the UP3 profile, type `Profile.clear()` then rerun the `@profile` line for UP3, then `ProfileView.view()` again.
""")