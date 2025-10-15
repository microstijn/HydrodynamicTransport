# debug_curvi_nans.jl

using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using Test
using UnicodePlots

# Explicitly import the internal functions we'll be using
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.VectorOperationsModule: rotate_velocities_to_grid!

println("--- NaN Debugging Script for Curvilinear Transport ---")

# --- 1. Programmatically Generate a Simple Curvilinear Grid ---
NG = 2
nx, ny, nz = 80, 50, 1 # A simple 2D case is best for debugging this
Lx, Ly = 400.0, 200.0
dx, dy, dz = Lx/nx, Ly/ny, 10.0
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG


# Allocate arrays for all grid metrics
lon_rho = zeros(Float64, nx_tot, ny_tot)
lat_rho = zeros(Float64, nx_tot, ny_tot)
angle   = zeros(Float64, nx_tot, ny_tot)
pm      = fill(1.0 / dx, nx_tot, ny_tot)
pn      = fill(1.0 / dy, nx_tot, ny_tot)
h       = fill(dz, nx_tot, ny_tot) # Constant depth

# Create a gentle bend in the grid within the physical domain
for j_phys in 1:ny, i_phys in 1:nx
    i_glob, j_glob = i_phys + NG, j_phys + NG
    x = (i_phys - 0.5) * dx
    y = (j_phys - 0.5) * dy
    
    bend_factor = 0.1 * sin(pi * x / Lx)
    lon_rho[i_glob, j_glob] = x
    lat_rho[i_glob, j_glob] = y + Ly * bend_factor
    # Angle is the derivative of the bend
    angle[i_glob, j_glob] = -atan(Ly * bend_factor * (pi / Lx) * cos(pi * x / Lx))
end

# Extrapolate metrics into ghost cells (simple constant extrapolation)
extrapolate!(A) = begin
    for j in (NG+1):(ny+NG); A[1:NG, j] .= A[NG+1, j]; A[(nx+NG+1):nx_tot, j] .= A[nx+NG, j]; end
    for i in 1:nx_tot; A[i, 1:NG] .= A[i, NG+1]; A[i, (ny+NG+1):ny_tot] .= A[i, ny+NG]; end
end
for arr in [lon_rho, lat_rho, angle, h]; extrapolate!(arr); end

# Calculate volumes and face areas
volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)
face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)

grid = CurvilinearGrid(NG, nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, 
                       [-dz, 0.0], pm, pn, angle, h,
                       trues(nx_tot,ny_tot), trues(nx_tot+1,ny_tot), trues(nx_tot,ny_tot+1),
                       face_area_x, face_area_y, volume)


# --- 2. Initialize State and Uniform Hydrodynamics ---
state = initialize_state(grid, (:C,))
state.time = 0.0

# Define a constant GEOGRAPHIC flow field (1.0 m/s East, 0.0 m/s North)
u_east_const = ones(Float64, nx, ny, nz)
v_north_const = zeros(Float64, nx, ny, nz)

println("Projecting uniform geographic velocities onto the curvilinear grid...")
# This function correctly fills state.u and state.v with grid-aligned velocities
rotate_velocities_to_grid!(state.u, state.v, grid, u_east_const, v_north_const)


# --- 3. Set Initial Condition ---
C = state.tracers[:C]
# Place a square patch of tracer near the left side
start_i, end_i = 10, 20
start_j, end_j = div(ny,2) - 5, div(ny,2) + 5
C[(start_i+NG):(end_i+NG), (start_j+NG):(end_j+NG), 1] .= 100.0

println("\n--- Initial State (t=0) ---")
C_phys_view = view(C, (NG+1):(nx+NG), (NG+1):(ny+NG), 1)
using UnicodePlots
println(UnicodePlots.heatmap(C_phys_view', title="Tracer at t=0", colormap=:viridis, labels=false))


# --- 4. Run the Simulation Loop with NaN Checking ---
# Simulation parameters
# Courant Number = u*dt/dx approx 1.0 * dt / 5.0. For stability, dt should be < 5.0.
dt = 2.0 
end_time = 200.0
n_steps = round(Int, end_time / dt)

# Choose the advection scheme to test: :TVD, :UP3, or :ImplicitADI
advection_scheme_to_test = :TVD

# Simple open boundaries
bcs = [OpenBoundary(side=:West), OpenBoundary(side=:East), OpenBoundary(side=:South), OpenBoundary(side=:North)]

println("\nRunning simulation for $n_steps steps with dt=$dt and scheme=:$advection_scheme_to_test...")

for step in 1:n_steps
    state.time += dt
    
    # Standard sequence of operations
    apply_boundary_conditions!(state, grid, bcs)
    # Note: We don't update hydrodynamics here because they are constant.
    horizontal_transport!(state, grid, dt, advection_scheme_to_test, 0.0, bcs)
    vertical_transport!(state, grid, dt, Dict{Symbol, SedimentParams}()) # No sediments
    
    # --- CRITICAL DEBUGGING STEP: Check for NaNs ---
    if any(isnan, state.tracers[:C])
        @error "NaNs DETECTED in tracer field at step $step (t=$(state.time))!"
        # Optional: Save the state right before the error for inspection
        # using JLD2; JLD2.save_object("nan_state.jld2", state)
        break # Stop the simulation
    end

    # Periodically print the state
    if mod(step, 20) == 0 || step == n_steps
        C_phys_view = view(state.tracers[:C], (NG+1):(nx+NG), (NG+1):(ny+NG), 1)
        println(UnicodePlots.heatmap(C_phys_view', title="Tracer at t=$(round(state.time, digits=1))", colormap=:viridis, labels=false))
    end
end

# --- 5. Final Verification ---
println("\n--- Final State (t=$(state.time)) ---")
@test !any(isnan, state.tracers[:C])
println("✅ Test Passed: Simulation completed without generating NaN values.")

println("\nExpected Behavior:")
println("- The tracer patch should move smoothly from left to right, following the gentle curve of the grid.")
println("- The shape of the patch should be reasonably well-maintained (depending on the scheme).")
println("- If the script stops with a 'NaNs DETECTED' error, the instability occurred in the last computed step.")