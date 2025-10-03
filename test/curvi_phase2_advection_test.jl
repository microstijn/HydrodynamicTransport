# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using NCDatasets
using UnicodePlots

# Explicitly import the functions we are testing/using
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VectorOperationsModule: rotate_velocities_to_grid!

println("--- Simple Curvilinear Advection Debugging Script ---")

# --- 2. Programmatically Generate a Curvilinear Grid ---
const NG = 2
nx, ny, nz = 100, 100, 2
Lx, Ly = 400.0, 400.0
center_x, center_y = Lx / 2, Ly / 2

# Total dimensions including ghost cells
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

# Create arrays for all grid metrics
lon_rho = zeros(Float64, nx_tot, ny_tot)
lat_rho = zeros(Float64, nx_tot, ny_tot)
angle   = zeros(Float64, nx_tot, ny_tot)
pm      = zeros(Float64, nx_tot, ny_tot)
pn      = zeros(Float64, nx_tot, ny_tot)

# Create a swirl grid (physical domain first)
for j_phys in 1:ny, i_phys in 1:nx
    i_glob, j_glob = i_phys + NG, j_phys + NG
    
    x = (i_phys - 0.5) * Lx / nx
    y = (j_phys - 0.5) * Ly / ny
    
    dist_from_center = sqrt((x - center_x)^2 + (y - center_y)^2)
    swirl_angle = π / 6 * (dist_from_center / (Lx / 2)) # Milder swirl
    
    lon_rho[i_glob, j_glob] = center_x + (x - center_x) * cos(swirl_angle) - (y - center_y) * sin(swirl_angle)
    lat_rho[i_glob, j_glob] = center_y + (x - center_x) * sin(swirl_angle) + (y - center_y) * cos(swirl_angle)
    angle[i_glob, j_glob] = swirl_angle
    pm[i_glob, j_glob] = nx / Lx
    pn[i_glob, j_glob] = ny / Ly
end

# Extrapolate metrics into ghost cells
extrapolate!(A) = begin
    for j in NG+1:ny+NG; A[1:NG, j] .= A[NG+1, j]; A[nx+NG+1:nx_tot, j] .= A[nx+NG, j]; end
    for i in 1:nx_tot; A[i, 1:NG] .= A[i, NG+1]; A[i, ny+NG+1:ny_tot] .= A[i, ny+NG]; end
end
for arr in [lon_rho, lat_rho, angle, pm, pn]; extrapolate!(arr); end

# Correctly calculate volume and face areas
face_area_x = zeros(nx_tot + 1, ny_tot, nz)
face_area_y = zeros(nx_tot, ny_tot + 1, nz)
volume = zeros(nx_tot, ny_tot, nz)
dz = 1.0

for k in 1:nz
    for j in 1:ny_tot, i in 1:nx_tot
        volume[i,j,k] = (1/pm[i,j]) * (1/pn[i,j]) * dz
    end
    for j in 1:ny_tot, i in 1:nx_tot+1
        local_dy = (i > 1 && i <= nx_tot) ? 0.5 * (1/pn[i-1,j] + 1/pn[i,j]) : 1/pn[min(i, nx_tot), j]
        face_area_x[i,j,k] = local_dy * dz
    end
    for j in 1:ny_tot+1, i in 1:nx_tot
        local_dx = (j > 1 && j <= ny_tot) ? 0.5 * (1/pm[i,j-1] + 1/pm[i,j]) : 1/pm[i, min(j, ny_tot)]
        face_area_y[i,j,k] = local_dx * dz
    end
end

# The lon_u/v arrays are not used by the rotation function, so we can pass lon_rho as a dummy
grid = CurvilinearGrid(NG, nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, 
                           [-dz, 0.0], pm, pn, angle, pm,
                           trues(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx_tot,ny_tot),
                           face_area_x, face_area_y, volume)

# --- 3. Initialize State and a Simple Flow Field ---
state = initialize_state(grid, (:C,))
state.time = 0.0

# Define a constant GEOGRAPHIC flow field (at cell centers)
u_east = ones(Float64, nx, ny, nz)
v_north = zeros(Float64, nx, ny, nz)

println("Projecting geographic velocities onto the curvilinear grid...")
# This function correctly fills state.u and state.v directly
rotate_velocities_to_grid!(state.u, state.v, grid, u_east, v_north)


# --- 4. Set Initial Condition ---
C = state.tracers[:C]
C_phys = view(C, NG+1:nx+NG, NG+1:ny+NG, :)
C_phys[3:6, 8:12, 1] .= 100.0

println("\n--- 1. Initial State (t=0) ---")
println(heatmap(C_phys[:,:,1]', title="Tracer at t=0", colormap=:viridis))

# --- 5. Run the Simulation ---
dt = 0.5; n_steps = 200
bcs = [OpenBoundary(side=:West), OpenBoundary(side=:East), OpenBoundary(side=:South), OpenBoundary(side=:North)]

println("Running simulation for $n_steps steps...")

for step in 1:n_steps
    state.time += dt
    
    # Apply boundary conditions first
    apply_boundary_conditions!(state, grid, bcs)
    
    # Then call the single, optimized transport function
    horizontal_transport!(state, grid, dt)
    
    if mod(step, 20) == 0
        C_phys_view = view(state.tracers[:C], NG+1:nx+NG, NG+1:ny+NG, 1)
        println(heatmap(C_phys_view', title="Tracer at t=$(round(state.time, digits=1))", colormap=:viridis))
    end
end

println("\n--- 3. Final State (t=$(state.time)) ---")
println(heatmap(view(state.tracers[:C], NG+1:nx+NG, NG+1:ny+NG, 1)', title="Tracer at t=$(state.time)", colormap=:viridis))
println("\n")

println("Verification:")
println("- The tracer should move smoothly across the curved grid, generally to the East.")
println("- The shape should deform slightly due to the curved grid and numerical diffusion.")
println("- There should be no 'NaN' values or instabilities.")

println("\n--- Debugging Script Finished ---")