using Pkg

#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using Test

using HydrodynamicTransport.SourceSinkModule: source_sink_terms!

# 1. Setup a minimal test environment
NG = 2
nx, ny, nz = 10, 10, 1
Lx, Ly = 200.0, 200.0
dx, dy, dz = Lx / nx, Ly / ny, 1.0
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
pm = ones(Float64, nx_tot, ny_tot) ./ dx
pn = ones(Float64, nx_tot, ny_tot) ./ dy
h = ones(Float64, nx_tot, ny_tot) .* 10.0
volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)

# Dummy arrays for unused fields
zeros_arr = zeros(nx_tot, ny_tot)
trues_arr_rho = trues(nx_tot, ny_tot)
trues_arr_u = trues(nx_tot + 1, ny_tot)
trues_arr_v = trues(nx_tot, ny_tot + 1)
face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)
z_w = [-dz, 0.0]
grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                       z_w, pm, pn, zeros_arr, h,
                       trues_arr_rho, trues_arr_u, trues_arr_v,
                       face_area_x, face_area_y, volume)
state = initialize_state(grid, (:Tracer1,))



C = state.tracers[:Tracer1]
C_phys = view(C, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
C_phys[4:6, 4:6, 1] .= 100.0 # Initial patch of tracer

# 2. Define a simple decay function
decay_rate = 0.1
simple_decay_func(C, env, dt) = Dict(:Tracer1 => -decay_rate * C[:Tracer1]) # Note: dt is not used here, change is per step


interaction = FunctionalInteraction(
    affected_tracers = [:Tracer1],
    interaction_function = (C, env, dt) -> Dict(:Tracer1 => C[:Tracer1] + (1 * dt))
)
# 3. Call the function to be tested
dt = 1.0

source_sink_terms!(
    state, 
    grid, 
    Vector{PointSource}(), 
    [interaction], 
    0.0, 
    dt, 
    0.0
)
# 4. Assert the result
initial_concentration = 100
expected_concentration = initial_concentration - (decay_rate * initial_concentration * dt)
final_concentration = state.tracers[:Tracer1][:, :, :]







# base
NG = 2
nx, ny, nz = 10, 10, 1
Lx, Ly = 200.0, 200.0
dx, dy, dz = Lx / nx, Ly / ny, 1.0
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
pm = ones(Float64, nx_tot, ny_tot) ./ dx
pn = ones(Float64, nx_tot, ny_tot) ./ dy
h = ones(Float64, nx_tot, ny_tot) .* 10.0
volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)

# Dummy arrays for unused fields
zeros_arr = zeros(nx_tot, ny_tot)
trues_arr_rho = trues(nx_tot, ny_tot)
trues_arr_u = trues(nx_tot + 1, ny_tot)
trues_arr_v = trues(nx_tot, ny_tot + 1)
face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)
z_w = [-dz, 0.0]
grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                       z_w, pm, pn, zeros_arr, h,
                       trues_arr_rho, trues_arr_u, trues_arr_v,
                       face_area_x, face_area_y, volume)
state = initialize_state(grid, (:Tracer1,))



C = state.tracers[:Tracer1]
C_phys = view(C, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
C_phys[4:6, 4:6, 1] .= 100.0 # Initial patch of tracer
# 2. Define a simple decay function and interaction
decay_rate = 0.1
decay_interaction = FunctionalInteraction(
    affected_tracers = [:Tracer1],
    interaction_function = (C, env, dt) -> Dict(:Tracer1 => -decay_rate * C[:Tracer1] * dt)
)
# 3. Run a short simulation with the interaction
final_state = run_simulation(
    grid,
    state,
    Vector{PointSource}(),
    0.0, # start_time
    1.0, # end_time
    1.0; # dt
    functional_interactions = [decay_interaction]
)
# 4. Assert the result
final_concentration = final_state.tracers[:Tracer1][:, :, :]

# ------------------------
# 2 tracer interaction
#-------------------------

NG = 2
nx, ny, nz = 10, 10, 1
Lx, Ly = 200.0, 200.0
dx, dy, dz = Lx / nx, Ly / ny, 1.0
nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
pm = ones(Float64, nx_tot, ny_tot) ./ dx
pn = ones(Float64, nx_tot, ny_tot) ./ dy
h = ones(Float64, nx_tot, ny_tot) .* 10.0
volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)

# Dummy arrays for unused fields
zeros_arr = zeros(nx_tot, ny_tot)
trues_arr_rho = trues(nx_tot, ny_tot)
trues_arr_u = trues(nx_tot + 1, ny_tot)
trues_arr_v = trues(nx_tot, ny_tot + 1)
face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)
z_w = [-dz, 0.0]
grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                       z_w, pm, pn, zeros_arr, h,
                       trues_arr_rho, trues_arr_u, trues_arr_v,
                       face_area_x, face_area_y, volume)
state = initialize_state(grid, (:Tracer1, :Tracer2))

tss_phys_view = view(state.tss, NG+1:nx+NG, NG+1:ny+NG, :)
tss_phys_view .= 5.0  # mg/L or appropriate units

# Example: Set a uniform UVB value of 1.2 across the physical domain
uvb_phys_view = view(state.uvb, NG+1:nx+NG, NG+1:ny+NG, :)
uvb_phys_view .= 1.2 # W/m^2 or appropriate units

C1 = state.tracers[:Tracer1]
C1_phys = view(C1, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
C1_phys[4:6, 4:6, 1] .= 100.0 # Initial patch of dissolved virus

C2 = state.tracers[:Tracer2]
C2_phys = view(C2, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
C2_phys[4:6, 4:6, 1] .= 50 # Initial patch of sorbed virus


k_total = 0.1  # Total decay rate for dissolved virus (1/day)
k_sorbed = 0.05 # Decay rate for sorbed virus (1/day)
k_ads = 0.001  # Adsorption rate (L/mg/day)
k_des = 0.2    # Desorption rate (1/day)
v_settle = 0.5 # Settling velocity (m/day)
# Convert rates from per-day to per-second for the simulation
SECONDS_PER_DAY = 86400.0

# 3. --- Define the Interaction Function ---
function virus_dynamics(concentrations::Dict, environment, dt::Float64)
    # Extract current concentrations for readability
    C_dissolved = concentrations[:Tracer1]
    C_sorbed = concentrations[:Tracer2]
    
    # Extract environmental parameters
    # The `environment` object provides access to fields like TSS, depth, etc.
    TSS = environment.TSS
    H = environment.depth # Water depth (H)

    # --- Change in Dissolved Virus (Tracer1) ---
    # dC_dissolved/dt = -k_total*C_dissolved - k_ads*C_dissolved*TSS + k_des*C_sorbed
    change_dissolved = (
        - (k_total / SECONDS_PER_DAY) * C_dissolved
        - (k_ads / SECONDS_PER_DAY) * C_dissolved * TSS
        + (k_des / SECONDS_PER_DAY) * C_sorbed
    ) * dt

    # --- Change in Sorbed Virus (Tracer2) ---
    # dC_sorbed/dt = -k_sorbed*C_sorbed + k_ads*C_dissolved*TSS - k_des*C_sorbed - (v_settle/H)*C_sorbed
    # Ensure depth H is not zero to prevent division errors
    settling_term = (H > 1e-6) ? (v_settle / SECONDS_PER_DAY / H) * C_sorbed : 0.0
    
    change_sorbed = (
        - (k_sorbed / SECONDS_PER_DAY) * C_sorbed
        + (k_ads / SECONDS_PER_DAY) * C_dissolved * TSS
        - (k_des / SECONDS_PER_DAY) * C_sorbed
        - settling_term
    ) * dt

    # Return a dictionary with the calculated changes
    return Dict(:Tracer1 => change_dissolved, :Tracer2 => change_sorbed)
end

virus_interaction = FunctionalInteraction(
    affected_tracers = [:Tracer1, :Tracer2],
    interaction_function = virus_dynamics
)


final_state = run_simulation(
    grid,
    state,
    Vector{PointSource}(),
    0.0, # start_time
    1.0, # end_time
    1.0; # dt
    functional_interactions = [virus_interaction]
)

final_concentration = final_state.tracers[:Tracer1][:, :, :]
final_concentration = final_state.tracers[:Tracer2][:, :, :]