# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using Test
using LinearAlgebra: norm

# Explicitly import the functions we are testing
using HydrodynamicTransport.VectorOperationsModule: rotate_velocities_to_grid, rotate_velocities_to_geographic

println("--- Test Script for VectorOperationsModule ---")

# --- 2. Create a Simple Curvilinear Grid for the Test ---
const NG = 2
nx, ny, nz = 10, 10, 1
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

# Create a grid with a constant 45-degree rotation
angle_val = π/4 
angle = fill(angle_val, (nx_tot, ny_tot))

# Dummy metric arrays (not important for this test, but needed by the struct)
dummy_array() = zeros(nx_tot, ny_tot)
dummy_3d(x,y,z) = zeros(x,y,z)

grid = CurvilinearGrid(NG, nx, ny, nz, 
                       dummy_array(), dummy_array(), dummy_array(), dummy_array(), dummy_array(), dummy_array(), 
                       [-1.0, 0.0], dummy_array(), dummy_array(), angle, dummy_array(),
                       trues(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx_tot,ny_tot),
                       dummy_3d(nx_tot+1, ny_tot, nz), dummy_3d(nx_tot, ny_tot+1, nz), dummy_3d(nx_tot, ny_tot, nz))

println("Test grid created with a constant rotation angle of $(round(rad2deg(angle_val), digits=1)) degrees.")

# --- 3. Define the Initial Geographic Velocity Field ---
# A simple, constant eastward flow of 1.0 m/s
u_east_initial = ones(Float64, nx, ny, nz)
v_north_initial = zeros(Float64, nx, ny, nz)

println("\n--- Initial Geographic Velocities (at cell centers) ---")
println("u_east = 1.0, v_north = 0.0")


# --- 4. Perform the "Round-Trip" Test ---
println("\nStep 1: Rotating from Geographic to Grid-Aligned coordinates...")
# This function returns full staggered arrays, including ghost cells which should be zero
u_grid, v_grid = rotate_velocities_to_grid(grid, u_east_initial, v_north_initial)

# Calculate the expected grid-aligned components at a rho-point
# u_grid_rho = 1.0 * cos(-π/4) - 0.0 * sin(-π/4) = cos(π/4) ≈ 0.707
# v_grid_rho = 0.0 * cos(-π/4) + 1.0 * sin(-π/4) = -sin(π/4) ≈ -0.707
println("Expected grid-aligned u at cell center: ~$(round(cos(angle_val), digits=3))")
println("Expected grid-aligned v at cell center: ~$(round(-sin(angle_val), digits=3))")


println("\nStep 2: Rotating from Grid-Aligned back to Geographic coordinates...")
# This function takes the full staggered arrays and returns physical-sized rho-point arrays
u_east_final, v_north_final = rotate_velocities_to_geographic(grid, u_grid, v_grid)

println("\n--- 5. Verification ---")
# The final result should be almost identical to our initial state.
# We use isapprox to handle tiny floating-point inaccuracies.
@test isapprox(u_east_initial, u_east_final)
@test isapprox(v_north_initial, v_north_final, atol=1e-12)

# We can also check the numerical error
error_u = norm(u_east_initial - u_east_final) / norm(u_east_initial)
error_v = norm(v_north_initial - v_north_final) # Cannot divide by zero norm

println("\nRound-trip test successful!")
println("Relative error in u_east: ", error_u)
println("Absolute error in v_north: ", error_v)

println("\n--- Test Script Finished ---")