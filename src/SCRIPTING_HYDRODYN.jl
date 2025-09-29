using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots
using NCDatasets


run_all_tests();

run_integration_tests()

nx, ny, nz = 80, 80, 1
Lx, Ly = 100.0, 100.0
center_x, center_y = Lx / 2, Ly / 2

lon_rho = zeros(Float64, nx, ny); lat_rho = zeros(Float64, nx, ny); angle = zeros(Float64, nx, ny)
for j in 1:ny, i in 1:nx
    x = (i - 0.5) * Lx / nx; y = (j - 0.5) * Ly / ny
    dist_from_center = sqrt((x - center_x)^2 + (y - center_y)^2)
    swirl_angle = π / 4 * (dist_from_center / (Lx / 2))
    
    rotated_x = center_x + (x - center_x) * cos(swirl_angle) - (y - center_y) * sin(swirl_angle)
    rotated_y = center_y + (x - center_x) * sin(swirl_angle) + (y - center_y) * cos(swirl_angle)
    
    lon_rho[i, j] = rotated_x; lat_rho[i, j] = rotated_y; angle[i, j] = swirl_angle
end

pm = fill(nx / Lx, (nx, ny)); pn = fill(ny / Ly, (nx, ny)); z_w = [-1.0, 0.0]
h = fill(1.0, (nx, ny)); mask = trues(nx, ny)
face_area_x = fill((Ly / ny) * 1.0, (nx + 1, ny, nz))
face_area_y = fill((Lx / nx) * 1.0, (nx, ny + 1, nz))
volume = fill((Lx / nx) * (Ly / ny) * 1.0, (nx, ny, nz))

grid = CurvilinearGrid(nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, z_w, 
                       pm, pn, angle, h, mask, mask, mask, face_area_x, face_area_y, volume)
println("Curvilinear grid generated successfully.")

# --- 3. Initialize State (Velocities start at zero) ---
state = initialize_state(grid, (:Tracer,))
C = state.tracers[:Tracer]

# --- 4. Set Initial Condition for the Tracer ---
cone_radius = 15.0
cone_center_x = 50.0
cone_center_y = 75.0

for j in 1:ny, i in 1:nx
    dist = sqrt((lon_rho[i,j] - cone_center_x)^2 + (lat_rho[i,j] - cone_center_y)^2)
    C[i,j,1] = max(0.0, 100.0 * (1.0 - dist / cone_radius))
end

println("\nInitial State:")
println(heatmap(C[:,:,1]', title="Tracer at Time = 0.0", colormap=:viridis, width=60))
mass_initial = sum(C .* grid.volume)
println("Initial Mass: ", mass_initial)
println("Initial Max Concentration: ", maximum(C))

# --- 5. Run the Simulation ---
# The runner will now call our new placeholder function at each step to generate the vortex.
dt = 0.1
period = 200.0
start_time = 0.0
end_time = period
sources = Vector{PointSource}() 

results, timesteps = run_and_store_simulation(grid, state, sources, start_time, end_time, dt, period / 10)
println("\nSimulation complete.")

# --- 6. Display Results --

for p in results
    C = p.tracers[:Tracer]
    println(heatmap(C[:,:,1]', colormap=:viridis, width=60))
end