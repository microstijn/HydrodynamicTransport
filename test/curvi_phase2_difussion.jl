# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using UnicodePlots
# Explicitly import the internal functions we want to test
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: diffuse_x!, diffuse_y!

println("--- Simple Curvilinear Diffusion Debugging Script ---")

# --- 2. Programmatically Generate a Curvilinear Grid ---
const NG = 2
nx, ny, nz = 20, 20, 1
Lx, Ly = 100.0, 100.0
center_x, center_y = Lx / 2, Ly / 2

# Total dimensions including ghost cells
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

# Create arrays for all grid metrics
lon_rho = zeros(Float64, nx_tot, ny_tot); lat_rho = zeros(Float64, nx_tot, ny_tot)
angle   = zeros(Float64, nx_tot, ny_tot); pm      = zeros(Float64, nx_tot, ny_tot)
pn      = zeros(Float64, nx_tot, ny_tot)

# Create a swirl grid (physical domain first)
for j_phys in 1:ny, i_phys in 1:nx
    i_glob, j_glob = i_phys + NG, j_phys + NG
    x = (i_phys - 0.5) * Lx / nx; y = (j_phys - 0.5) * Ly / ny
    dist = sqrt((x - center_x)^2 + (y - center_y)^2); ang = π / 6 * (dist / (Lx / 2))
    lon_rho[i_glob, j_glob] = center_x + (x-center_x)*cos(ang) - (y-center_y)*sin(ang)
    lat_rho[i_glob, j_glob] = center_y + (x-center_x)*sin(ang) + (y-center_y)*cos(ang)
    angle[i_glob, j_glob] = ang; pm[i_glob, j_glob] = nx/Lx; pn[i_glob, j_glob] = ny/Ly
end

# Extrapolate metrics into ghost cells
extrapolate!(A) = begin
    for j in NG+1:ny+NG; A[1:NG, j] .= A[NG+1, j]; A[nx+NG+1:nx_tot, j] .= A[nx+NG, j]; end
    for i in 1:nx_tot; A[i, 1:NG] .= A[i, NG+1]; A[i, ny+NG+1:ny_tot] .= A[i, ny+NG]; end
end
for arr in [lon_rho, lat_rho, angle, pm, pn]; extrapolate!(arr); end

# Correctly calculate volume and face areas
face_area_x = zeros(nx_tot + 1, ny_tot, nz); face_area_y = zeros(nx_tot, ny_tot + 1, nz)
volume = zeros(nx_tot, ny_tot, nz); dz = 1.0
for k in 1:nz, j in 1:ny_tot, i in 1:nx_tot; volume[i,j,k] = (1/pm[i,j])*(1/pn[i,j])*dz; end
for k in 1:nz, j in 1:ny_tot, i in 1:nx_tot+1; face_area_x[i,j,k] = ((i>1 && i<=nx_tot) ? 0.5*(1/pn[i-1,j]+1/pn[i,j]) : 1/pn[min(i, nx_tot),j])*dz; end
for k in 1:nz, j in 1:ny_tot+1, i in 1:nx_tot; face_area_y[i,j,k] = ((j>1 && j<=ny_tot) ? 0.5*(1/pm[i,j-1]+1/pm[i,j]) : 1/pm[i,min(j,ny_tot)])*dz; end

grid = CurvilinearGrid(NG, nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, 
                       [-dz, 0.0], pm, pn, angle, pm, 
                       trues(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx_tot,ny_tot),
                       face_area_x, face_area_y, volume)

# --- 3. Initialize State and Set Zero Flow Field ---
state = initialize_state(grid, (:C,))
state.time = 0.0

# Set all velocities to ZERO to isolate diffusion
state.u .= 0.0
state.v .= 0.0
state.w .= 0.0

# --- 4. Set Initial Condition ---
C = state.tracers[:C]
# Place a single "spike" of tracer in the center of the physical domain
center_i_phys, center_j_phys = div(nx, 2), div(ny, 2)
C[center_i_phys + NG, center_j_phys + NG, 1] = 100.0

println("\n--- 1. Initial State (t=0) ---")
println(heatmap(view(C, NG+1:nx+NG, NG+1:ny+NG, 1)', title="Tracer at t=0", colormap=:viridis))

# --- 5. Run the Simulation ---
Kh = 1.0
min_dx = 1.0 / maximum(pm); min_dy = 1.0 / maximum(pn)
min_spacing_sq = min(min_dx^2, min_dy^2)
dt = 0.1 * min_spacing_sq / (2*Kh) # Use a safe time step
n_steps = 200

bcs = [OpenBoundary(side=:West), OpenBoundary(side=:East), OpenBoundary(side=:South), OpenBoundary(side=:North)]

println("Running simulation for $n_steps steps (DIFFUSION ONLY)...")

# Calculate initial mass
mass_initial = sum(view(state.tracers[:C], NG+1:nx+NG, NG+1:ny+NG, :) .* view(grid.volume, NG+1:nx+NG, NG+1:ny+NG, :))

for step in 1:n_steps
    state.time += dt
    apply_boundary_conditions!(state, grid, bcs)
    
    C_current = state.tracers[:C]
    C_in_step = deepcopy(C_current)
    C_temp = deepcopy(C_current)

    diffuse_x!(C_temp, C_in_step, grid, dt, Kh)
    copyto!(C_in_step, C_temp)
    diffuse_y!(C_current, C_in_step, grid, dt, Kh)
    
    if mod(step, 10) == 0
        println(heatmap(view(C_current, NG+1:nx+NG, NG+1:ny+NG, 1)', title="Tracer at t=$(state.time)", colormap=:viridis))
    end
end

println("\n--- 3. Final State (t=$(state.time)) ---")
println(heatmap(view(state.tracers[:C], NG+1:nx+NG, NG+1:ny+NG, 1)', title="Tracer at t=$(state.time)", colormap=:viridis))
println("\n")

# --- 6. Verification ---
mass_final = sum(view(state.tracers[:C], NG+1:nx+NG, NG+1:ny+NG, :) .* view(grid.volume, NG+1:nx+NG, NG+1:ny+NG, :))

println("Verification:")
println("- The tracer should have spread out from the center.")
println("- The pattern may look distorted due to the curved grid, which is correct.")
println("- There should be no 'NaN' values or instabilities.")
println("- The total mass must be conserved due to no-flux boundaries.")
println("  Initial Mass: ", mass_initial)
println("  Final Mass:   ", mass_final)
@assert isapprox(mass_initial, mass_final, rtol=1e-9) "Mass conservation failed!"
println("✅ Mass conservation test passed.")

println("\n--- Debugging Script Finished ---")