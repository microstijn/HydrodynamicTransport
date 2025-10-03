# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HydrodynamicTransport
using NCDatasets
using Test
using UnicodePlots

println("--- Vertical Transport Debugging Script ---")

# ==============================================================================
println("\n\n--- PART 1: Testing on a Cartesian Grid ---")
# ==============================================================================

# --- Setup for Cartesian Test ---
const NG = 2
nx_cart, ny_cart, nz_cart = 1, 1, 20
grid_cart = initialize_cartesian_grid(nx_cart, ny_cart, nz_cart, 10.0, 10.0, 100.0; ng=NG)
state_cart = initialize_state(grid_cart, (:C,))

# Set all velocities to zero initially
state_cart.u .= 0.0
state_cart.v .= 0.0
state_cart.w .= 0.0

# Define a constant upward velocity in the single physical column
const W_VELOCITY = 0.01 # m/s
i_phys_cart, j_phys_cart = 1, 1
i_glob_cart, j_glob_cart = i_phys_cart + NG, j_phys_cart + NG
state_cart.w[i_glob_cart, j_glob_cart, :] .= W_VELOCITY

# Initial condition: a thin layer of tracer in the middle
C_cart = state_cart.tracers[:C]
start_k = div(nz_cart, 2)
C_cart[i_glob_cart, j_glob_cart, start_k:start_k+1] .= 100.0

println("\n--- Initial State (Cartesian) ---")
z_centers_cart = [grid_cart.z[i_glob_cart, j_glob_cart, k] for k in 1:nz_cart]
tracer_profile_cart_initial = C_cart[i_glob_cart, j_glob_cart, :]
println(lineplot(z_centers_cart, tracer_profile_cart_initial, title="Initial Tracer Profile", xlabel="Depth (m)", ylabel="Concentration"))

# --- Run Simulation (Cartesian) ---
dt_cart = 20.0
n_steps_cart = 50
println("Running simulation for $n_steps_cart steps...")

for _ in 1:n_steps_cart
    # Note: No boundary conditions needed as we are only testing the vertical module on a single column
    vertical_transport!(state_cart, grid_cart, dt_cart)
end

# --- Verification (Cartesian) ---
println("\n--- Final State (Cartesian) ---")
tracer_profile_cart_final = state_cart.tracers[:C][i_glob_cart, j_glob_cart, :]
println(lineplot(z_centers_cart, tracer_profile_cart_final, title="Final Tracer Profile", xlabel="Depth (m)", ylabel="Concentration"))

# 1. Check for upward movement
peak_k_initial = argmax(tracer_profile_cart_initial)
peak_k_final = argmax(tracer_profile_cart_final)
@test peak_k_final > peak_k_initial
println("\n✅ Verification Passed: Tracer peak moved upward from index $peak_k_initial to $peak_k_final.")

# 2. Check for mass conservation
mass_initial_cart = sum(tracer_profile_cart_initial)
mass_final_cart = sum(tracer_profile_cart_final)
@test isapprox(mass_initial_cart, mass_final_cart, rtol=1e-9)
println("✅ Verification Passed: Mass in the column was conserved.")


# ==============================================================================
println("\n\n--- PART 2: Testing on a Simple Curvilinear Grid ---")
# ==============================================================================

# --- Setup for Simple Curvilinear Test ---
nx_curvi, ny_curvi, nz_curvi = 1, 1, 20
nx_tot, ny_tot = nx_curvi + 2*NG, ny_curvi + 2*NG

# Create simple metric arrays with a constant angle
angle = fill(π/4, (nx_tot, ny_tot))
pm    = fill(1/10.0, (nx_tot, ny_tot)) # 10m grid spacing
pn    = fill(1/10.0, (nx_tot, ny_tot))

# Define the vertical grid
z_w_curvi = collect(range(-100.0, 0.0, length=nz_curvi+1))
dz = z_w_curvi[2] - z_w_curvi[1]

# Create dummy arrays for other grid fields
dummy_2d() = zeros(nx_tot, ny_tot)
dummy_3d(x,y,z) = zeros(x,y,z)
dummy_bool() = trues(nx_tot, ny_tot)
volume = fill((1/pm[1,1]) * (1/pn[1,1]) * dz, (nx_tot, ny_tot, nz_curvi))

grid_curvi = CurvilinearGrid(NG, nx_curvi, ny_curvi, nz_curvi,
                           dummy_2d(), dummy_2d(), dummy_2d(), dummy_2d(), dummy_2d(), dummy_2d(),
                           z_w_curvi, pm, pn, angle, dummy_2d(),
                           dummy_bool(), dummy_bool(), dummy_bool(),
                           dummy_3d(nx_tot+1, ny_tot, nz_curvi), dummy_3d(nx_tot, ny_tot+1, nz_curvi), volume)

state_curvi = initialize_state(grid_curvi, (:C,))
state_curvi.u .= 0.0; state_curvi.v .= 0.0; state_curvi.w .= 0.0

# Define a constant upward velocity in the single physical column
i_phys_curvi, j_phys_curvi = 1, 1
i_glob_curvi, j_glob_curvi = i_phys_curvi + NG, j_phys_curvi + NG
state_curvi.w[i_glob_curvi, j_glob_curvi, :] .= W_VELOCITY

# Initial condition
C_curvi = state_curvi.tracers[:C]
start_k_curvi = div(nz_curvi, 2)
C_curvi[i_glob_curvi, j_glob_curvi, start_k_curvi:start_k_curvi+1] .= 100.0

println("\n--- Initial State (Curvilinear) ---")
z_centers_curvi = 0.5 .* (z_w_curvi[1:end-1] .+ z_w_curvi[2:end])
tracer_profile_curvi_initial = C_curvi[i_glob_curvi, j_glob_curvi, :]
println(lineplot(z_centers_curvi, tracer_profile_curvi_initial, title="Initial Tracer Profile (Curvilinear)", xlabel="Depth (m)", ylabel="Concentration"))

# --- Run Simulation (Curvilinear) ---
dt_curvi = 20.0
n_steps_curvi = 50
println("Running simulation for $n_steps_curvi steps...")

for _ in 1:n_steps_curvi
    vertical_transport!(state_curvi, grid_curvi, dt_curvi)
end

# --- Verification (Curvilinear) ---
println("\n--- Final State (Curvilinear) ---")
tracer_profile_curvi_final = state_curvi.tracers[:C][i_glob_curvi, j_glob_curvi, :]
println(lineplot(z_centers_curvi, tracer_profile_curvi_final, title="Final Tracer Profile (Curvilinear)", xlabel="Depth (m)", ylabel="Concentration"))

peak_k_initial_curvi = argmax(tracer_profile_curvi_initial)
peak_k_final_curvi = argmax(tracer_profile_curvi_final)
@test peak_k_final_curvi > peak_k_initial_curvi
println("\n✅ Verification Passed: Tracer peak moved upward from index $peak_k_initial_curvi to $peak_k_final_curvi.")

mass_initial_curvi = sum(tracer_profile_curvi_initial)
mass_final_curvi = sum(tracer_profile_curvi_final)
@test isapprox(mass_initial_curvi, mass_final_curvi, rtol=1e-9)
println("✅ Verification Passed: Mass in the column was conserved.")


println("\n--- Debugging Script Finished ---")
