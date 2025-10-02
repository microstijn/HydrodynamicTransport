# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using UnicodePlots
using Test
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.SourceSinkModule: source_sink_terms!
using HydrodynamicTransport.VectorOperationsModule: rotate_velocities_to_grid!


println("--- Tidal System with Point Sources Test Script ---")

# --- 2. Main Configuration ---
# Set this flag to 'true' to run on a simple Cartesian grid,
# or 'false' to run on the programmatically generated Curvilinear grid.
USE_CARTESIAN_GRID = false

# --- 3. Grid and State Setup ---
NG = 2
nx, ny, nz = 80, 50, 1 # A 2D simulation is sufficient for this test

# Define two tracers for this simulation
TRACER_NAMES = (:Pollutant, :Salinity)

local grid, state # Use local to ensure these are accessible outside the if/else block

if USE_CARTESIAN_GRID
    println("Setting up a CARTESIAN grid...")
    grid = initialize_cartesian_grid(nx, ny, nz, 400.0, 200.0, 10.0; ng=NG)
else
    println("Setting up a CURVILINEAR grid...")
    # This section programmatically creates a curved grid for the test
    Lx, Ly = 400.0, 200.0
    center_x, center_y = Lx / 2, Ly / 2
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

    lon_rho = zeros(Float64, nx_tot, ny_tot)
    lat_rho = zeros(Float64, nx_tot, ny_tot)
    angle   = zeros(Float64, nx_tot, ny_tot)
    pm      = fill(nx / Lx, nx_tot, ny_tot)
    pn      = fill(ny / Ly, nx_tot, ny_tot)
    
    for j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + NG, j_phys + NG
        x = (i_phys - 0.5) * Lx / nx; y = (j_phys - 0.5) * Ly / ny
        # Create a gentle bend in the grid
        bend_factor = 0.1 * sin(pi * x / Lx)
        lon_rho[i_glob, j_glob] = x
        lat_rho[i_glob, j_glob] = y + Ly * bend_factor
        angle[i_glob, j_glob] = -atan(Ly * bend_factor * (pi / Lx) * cos(pi * x / Lx))
    end
    
    extrapolate!(A) = begin
        for j in NG+1:ny+NG; A[1:NG, j] .= A[NG+1, j]; A[nx+NG+1:nx_tot, j] .= A[nx+NG, j]; end
        for i in 1:nx_tot; A[i, 1:NG] .= A[i, NG+1]; A[i, ny+NG+1:ny_tot] .= A[i, ny+NG]; end
    end
    for arr in [lon_rho, lat_rho, angle]; extrapolate!(arr); end

    face_area_x = ones(nx_tot + 1, ny_tot, nz) .* (Ly / ny)
    face_area_y = ones(nx_tot, ny_tot + 1, nz) .* (Lx / nx)
    volume = ones(nx_tot, ny_tot, nz) .* (Lx/nx * Ly/ny)

    grid = CurvilinearGrid(NG, nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, 
                           [-1.0, 0.0], pm, pn, angle, pm,
                           trues(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx_tot,ny_tot),
                           face_area_x, face_area_y, volume)
end

state = initialize_state(grid, TRACER_NAMES)

# --- 4. Define Tidal Hydrodynamics ---
const TIDAL_PERIOD = 300.0 # seconds
const MAX_VELOCITY = 0.5   # m/s

# We use multiple dispatch to handle the different grid types
function update_tidal_hydrodynamics!(state::State, grid::CartesianGrid)
    omega = 2π / TIDAL_PERIOD
    current_u = MAX_VELOCITY * cos(omega * state.time)
    state.u .= current_u # Simple, uniform oscillating flow
    state.v .= 0.0
end

function update_tidal_hydrodynamics!(state::State, grid::CurvilinearGrid)
    omega = 2π / TIDAL_PERIOD
    # Define a geographic East-West oscillating flow
    u_east_val = MAX_VELOCITY * cos(omega * state.time)
    u_east = fill(u_east_val, (nx, ny, nz))
    v_north = zeros(Float64, nx, ny, nz)
    # Project this geographic flow onto the curved grid
    rotate_velocities_to_grid!(state.u, state.v, grid, u_east, v_north)
end


# --- 5. Define Boundary Conditions and Point Sources ---
sources = [
    PointSource(i=20, j=25, k=1, tracer_name=:Pollutant, influx_rate=(t)->50.0),
    PointSource(i=60, j=15, k=1, tracer_name=:Pollutant, influx_rate=(t)->25.0),
    PointSource(i=60, j=35, k=1, tracer_name=:Pollutant, influx_rate=(t)->25.0)
]

# --- FIX: This function should only define the composition of external water. ---
# The BoundaryConditionsModule will use this dictionary ONLY when there is inflow.
function tidal_inflow_concentrations(time::Float64)
    # This dictionary defines the concentration of tracers in the ocean.
    return Dict(:Pollutant => 0.0, :Salinity => 35.0)
end

boundary_conditions = [
    TidalBoundary(side=:East, inflow_concentrations=tidal_inflow_concentrations),
    OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)
]

# --- 6. Run the Simulation ---
dt = 1.0
start_time = 0.0
end_time = TIDAL_PERIOD * 1.5 # Run for 1.5 tidal cycles

println("\nRunning simulation for $(end_time)s with dt=$(dt)s...")
n_steps = round(Int, (end_time - start_time) / dt)

for step in 1:n_steps
    global state # Ensure state is updated in the global scope of the script
    state.time += dt

    # The order of operations is critical
    apply_boundary_conditions!(state, grid, boundary_conditions)
    update_tidal_hydrodynamics!(state, grid)
    horizontal_transport!(state, grid, dt)
    vertical_transport!(state, grid, dt) # Will do nothing for nz=1, but good practice
    source_sink_terms!(state, grid, sources, state.time, dt)

    # --- Plotting ---
    if mod(step, round(Int, n_steps / 10)) == 0 || step == n_steps
        println("\n--- Time: $(round(state.time, digits=1))s ---")
        
        for name in TRACER_NAMES
            tracer_phys = view(state.tracers[name], NG+1:nx+NG, NG+1:ny+NG, 1)
            println(heatmap(tracer_phys', title="$name Concentration", colormap=:viridis, labels=false))
        end
        # Print current tidal velocity
        u_vel = state.u[div(size(state.u,1),2), div(size(state.u,2),2), 1]
        phase = u_vel > 0 ? "Flood (Inflow ->)" : "Ebb (Outflow <-)" # Note: For East boundary, u>0 is Ebb
        println("Tidal Phase: $phase (u ≈ $(round(u_vel, digits=2)) m/s)")
    end
end

# --- 7. Verification ---
println("\n--- Final Verification ---")
@test !any(isnan, state.tracers[:Pollutant])
@test !any(isnan, state.tracers[:Salinity])
println("✅ Test Passed: Simulation completed without generating NaN values.")
println("\nExpected Behavior:")
println("- Plumes should form at the point source locations and be carried by the tidal flow.")
println("- During flood tide (negative velocity), the plumes should move west (left) and high salinity water should enter from the east (right).")
println("- During ebb tide (positive velocity), the plumes and low-salinity water should be flushed out to the east.")
println("\n--- Test Script Finished ---")