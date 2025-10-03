# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using UnicodePlots

println("--- Advection Scheme Comparison Test ---")

# --- 2. Configuration ---
USE_CARTESIAN_GRID = true # Set to false to use a simple curvilinear grid

# --- Grid & Simulation Parameters ---
nx, ny, nz = 80, 40, 1
ng = 2 # Number of ghost cells defined in the library
dt = 0.4
end_time = 40.0

# ==============================================================================
# --- 3. Setup Functions ---
# ==============================================================================

function setup_grid()
    if USE_CARTESIAN_GRID
        println("Using Cartesian Grid")
        return initialize_cartesian_grid(nx, ny, nz, 80.0, 40.0, 10.0; ng=ng)
    else
        println("Using Curvilinear Grid")
        # Create a simple sheared grid
        nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
        lon_rho = zeros(nx_tot, ny_tot); lat_rho = zeros(nx_tot, ny_tot)
        angle = zeros(nx_tot, ny_tot); pm = ones(nx_tot, ny_tot); pn = ones(nx_tot, ny_tot)
        
        shear_factor = 0.5
        for j in 1:ny_tot, i in 1:nx_tot
            lon_rho[i,j] = i - ng
            lat_rho[i,j] = (j - ng) + shear_factor * (i - ng)
        end
        
        return CurvilinearGrid(ng, nx, ny, nz, 
            lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, [-1.0, 0.0], 
            pm, pn, angle, pm, trues(nx_tot,ny_tot), trues(nx_tot,ny_tot), trues(nx_tot,ny_tot),
            ones(nx_tot+1, ny_tot, nz), ones(nx_tot, ny_tot+1, nz), ones(nx_tot, ny_tot, nz))
    end
end

function setup_initial_state(grid::AbstractGrid)
    # Initialize state with one tracer
    state = initialize_state(grid, (:C,))

    # Create a sharp square pulse of tracer
    start_x, end_x = round(Int, nx/4), round(Int, nx/2)
    start_y, end_y = round(Int, ny/4), round(Int, ny*3/4)
    state.tracers[:C][start_x+ng:end_x+ng, start_y+ng:end_y+ng, :] .= 100.0

    # Define a constant, diagonal velocity field
    u_geo = 1.0; v_geo = 0.5
    
    if isa(grid, CurvilinearGrid)
        u_east = fill(u_geo, (nx, ny, nz))
        v_north = fill(v_geo, (nx, ny, nz))
        rotate_velocities_to_grid!(state.u, state.v, grid, u_east, v_north)
    else # CartesianGrid
        state.u .= u_geo
        state.v .= v_geo
    end
    
    return state
end

function plot_state(state::State, grid::AbstractGrid, title::String)
    tracer_phys = view(state.tracers[:C], ng+1:nx+ng, ng+1:ny+ng, 1)
    println("\n--- ", title, " ---")
    println(heatmap(tracer_phys', colormap=:viridis, labels=false, title=title))
end


# ==============================================================================
# --- 4. Main Execution ---
# ==============================================================================

# --- Setup ---
grid = setup_grid()
initial_state = setup_initial_state(grid)

# Define empty sources and simple open boundaries for the simulation
sources = Vector{PointSource}()
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

# --- Plot Initial State ---
plot_state(initial_state, grid, "Initial State (t=0)")

# --- Run Simulation with TVD Scheme ---
println("\nRunning simulation with default :TVD scheme...")
state_tvd = deepcopy(initial_state) # Ensure a fresh start
final_state_tvd = run_simulation(grid, state_tvd, sources, 0.0, end_time, dt; boundary_conditions=bcs, advection_scheme=:TVD)

# --- Run Simulation with UP3 Scheme ---
println("\nRunning simulation with :UP3 scheme...")
state_up3 = deepcopy(initial_state) # Ensure a fresh start
final_state_up3 = run_simulation(grid, state_up3, sources, 0.0, end_time, dt; boundary_conditions=bcs, advection_scheme=:UP3)

# --- Plot Final Results for Comparison ---
plot_state(final_state_tvd, grid, "Final State with TVD Scheme")
plot_state(final_state_up3, grid, "Final State with UP3 Scheme")

println("\n--- Comparison ---")
println("Observe the difference in the shape of the tracer pulse.")
println("The :UP3 scheme should preserve the sharp corners and edges more effectively,")
println("while the :TVD scheme may appear more smeared or 'diffused'.")
println("\n--- Test Script Finished ---")