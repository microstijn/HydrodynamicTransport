# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using BenchmarkTools

# We need to explicitly import the function we want to benchmark
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!

println("--- Advection Scheme Speed Comparison ---")

# ==============================================================================
# --- 2. Setup a Realistic Test Case ---
# ==============================================================================

println("Setting up test grid and state...")

# A larger grid will give more meaningful performance results
nx, ny, nz = 200, 200, 5
ng = 2

# Initialize grid and state
grid = initialize_cartesian_grid(nx, ny, nz, 2000.0, 2000.0, 50.0; ng=ng)
state = initialize_state(grid, (:C,))

# Fill the tracer with some non-uniform data
for k in 1:nz, j in 1:ny+2*ng, i in 1:nx+2*ng
    state.tracers[:C][i,j,k] = sin(i/10.0) * cos(j/10.0)
end

# Set a constant velocity field so the advection functions have work to do
state.u .= 1.0
state.v .= 0.5

# A stable timestep for this setup
dt = 0.2

println("Setup complete. Beginning benchmarks...")

# ==============================================================================
# --- 3. Run Benchmarks ---
# ==============================================================================

# Note: The `$` before the variables is important for BenchmarkTools.
# It ensures the variables are correctly interpolated into the benchmark expression
# without measuring the cost of accessing global variables.

println("\n--- Benchmarking :TVD Scheme (Default) ---")
# The first run will include compilation time. BenchmarkTools automatically handles this.
@btime horizontal_transport!($state, $grid, $dt, :TVD)

println("\n--- Benchmarking :UP3 Scheme (New) ---")
@btime horizontal_transport!($state, $grid, $dt, :UP3)


# ==============================================================================
# --- 4. Analysis ---
# ==============================================================================

println("\n--- Analysis of Results ---")
println("The benchmark results show the median time, memory allocation, and garbage collection (gc) time for one call to `horizontal_transport!`.")
println("\nAs predicted, the :UP3 scheme should be significantly faster and allocate less memory than the more complex :TVD scheme.")
println("This demonstrates the trade-off: :UP3 is faster, while :TVD is designed to be more robust against oscillations (though often more diffusive).")

println("\n--- Speed Comparison Finished ---")