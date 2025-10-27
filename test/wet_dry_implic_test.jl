# test/test_wetting_drying.jl

using Test
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule # For manual grid
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.HydrodynamicsModule # For placeholder state setup
using HydrodynamicTransport.HorizontalTransportModule # Functions to test
using HydrodynamicTransport.FluxLimitersModule

println("\n--- Running Wetting/Drying Test Script (Curvilinear Grid) ---")

# ==============================================================================
# --- 1. Setup Grid with a Dry Island ---
# ==============================================================================
nx, ny, nz = 20, 20, 1 # 2D focus is sufficient for horizontal test
Lx, Ly, Lz = 20.0, 20.0, 10.0
dx = Lx / nx # 1.0
dy = Ly / ny # 1.0
ng = 2

D_crit = 0.1 # Critical depth for wet/dry

# --- Manually construct CurvilinearGrid ---
nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

pm_full = fill(1.0 / dx, (nx_tot, ny_tot))
pn_full = fill(1.0 / dy, (nx_tot, ny_tot))
angle_full = zeros(nx_tot, ny_tot)
z_w_vec = [-Lz, 0.0]
dz = Lz # Only one layer

# Bathymetry (h) - Make a dry island in the center
h_full = fill(5.0, (nx_tot, ny_tot)) # Base depth 5.0
island_i_start, island_i_end = ng + 8, ng + 12
island_j_start, island_j_end = ng + 8, ng + 12
h_full[island_i_start:island_i_end, island_j_start:island_j_end] .= -0.1 # Island base slightly below 0

mask_rho_full = h_full .> 0.0 # Mask based on h > 0 initially
mask_u_full = ones(Bool, nx_tot + 1, ny_tot) # Will be effectively handled by D_crit
mask_v_full = ones(Bool, nx_tot, ny_tot + 1) # Will be effectively handled by D_crit

lon_rho, lat_rho = zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot)
lon_u, lat_u = zeros(nx_tot + 1, ny_tot), zeros(nx_tot + 1, ny_tot)
lon_v, lat_v = zeros(nx_tot, ny_tot + 1), zeros(nx_tot, ny_tot + 1)

face_area_x = fill(dy * dz, (nx_tot + 1, ny_tot, nz))
face_area_y = fill(dx * dz, (nx_tot, ny_tot + 1, nz))
volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))
# Zero out volume on the island based on h
volume[h_full .<= 0.0, 1] .= 0.0 # For nz=1

grid = CurvilinearGrid(
    ng, nx, ny, nz,
    lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
    z_w_vec,
    pm_full, pn_full, angle_full, h_full,
    mask_rho_full, mask_u_full, mask_v_full,
    face_area_x, face_area_y, volume
)

# --- State Initialization ---
state = initialize_state(grid, (:C,))

# Set Zeta such that the island is dry (Total Depth H = h + zeta < D_crit)
state.zeta .= 0.0 # Start with flat surface
# Total depth on island = -0.1 + 0.0 = -0.1, which is < D_crit (0.1) -> DRY
# Total depth off island = 5.0 + 0.0 = 5.0, which is > D_crit -> WET

# --- Initial Condition: Tracer patch next to the island ---
patch_i_start, patch_i_end = ng + 5, ng + 7 # Cells i=5,6,7
patch_j_start, patch_j_end = ng + 8, ng + 12 # Cells j=8,9,10,11,12
state.tracers[:C][patch_i_start:patch_i_end, patch_j_start:patch_j_end, 1] .= 1.0

println("Initial State Setup:")
println(" D_crit = $D_crit")
println(" Island physical indices: i=$(island_i_start-ng):$(island_i_end-ng), j=$(island_j_start-ng):$(island_j_end-ng)")
println(" Initial tracer patch: i=$(patch_i_start-ng):$(patch_i_end-ng), j=$(patch_j_start-ng):$(patch_j_end-ng)")
println(" Total Depth on island = $(h_full[island_i_start, island_j_start] + state.zeta[island_i_start, island_j_start, 1])")
println(" Total Depth off island = $(h_full[ng+1, ng+1] + state.zeta[ng+1, ng+1, 1])")
println(" Max initial tracer on island = ", maximum(state.tracers[:C][island_i_start:island_i_end, island_j_start:island_j_end, 1]))


# --- Hydrodynamics: Flow towards the island (positive u) ---
u_val = 1.0
state.u .= u_val
state.v .= 0.0 # No v-flow for simplicity
state.w .= 0.0

# --- Parameters ---
dt = 0.5 * dx / u_val # CFL = 0.5 based on dx
Kh = 0.0
limiter_func = van_leer
n_steps = 3 # Advect for 3 steps, should try to cross onto island

# Buffers
C_in = state.tracers[:C]
C_buffer1 = state._buffer1[:C]
C_buffer2 = C_in # Use C_in as the final buffer for y-sweep output

# ==============================================================================
# --- 2. Run Advection Steps ---
# ==============================================================================
println("\nRunning $n_steps advection steps (dt=$dt)...")

C_current_x = C_in
C_intermediate_y = C_buffer1
C_final = C_buffer2 # Will point to C_in after loop

for step in 1:n_steps
    println(" Step $step:")
    # X-Sweep: Input C_current_x, Output C_intermediate_y
    advect_diffuse_tvd_implicit_x!(C_intermediate_y, C_current_x, state, grid, dt, Kh, limiter_func, D_crit)

    # Check for negatives or mass entering dry zone after X
    max_on_island_x = maximum(C_intermediate_y[island_i_start:island_i_end, island_j_start:island_j_end, 1])
    min_after_x = minimum(C_intermediate_y)
    println("   After X: Min C = $(round(min_after_x, digits=4)), Max C on Island = $(round(max_on_island_x, digits=4))")
    @test max_on_island_x <= 1e-12 # Should be essentially zero
    @test min_after_x >= -1e-9

    # Y-Sweep: Input C_intermediate_y, Output C_final
    advect_diffuse_tvd_implicit_y!(C_final, C_intermediate_y, state, grid, dt, Kh, limiter_func, D_crit)

    # Check for negatives or mass entering dry zone after Y
    max_on_island_y = maximum(C_final[island_i_start:island_i_end, island_j_start:island_j_end, 1])
    min_after_y = minimum(C_final)
    println("   After Y: Min C = $(round(min_after_y, digits=4)), Max C on Island = $(round(max_on_island_y, digits=4))")
    @test max_on_island_y <= 1e-12 # Should be essentially zero
    @test min_after_y >= -1e-9

    # Prepare for next step (no need to swap if output is C_in)
    # C_current_x = C_final
end

# ==============================================================================
# --- 3. Final Check ---
# ==============================================================================
println("\n--- Final State Check ---")
final_C = state.tracers[:C] # This is C_final because we used C_in as the last output buffer
final_max_on_island = maximum(final_C[island_i_start:island_i_end, island_j_start:island_j_end, 1])
final_min = minimum(final_C)

println("Final Max C on Island: $(round(final_max_on_island, digits=6))")
println("Final Min C Overall: $(round(final_min, digits=6))")

@testset "Wetting/Drying Final Check" begin
    @test final_max_on_island <= 1e-12
    @test final_min >= -1e-9
end

println("\n--- Wetting/Drying Test Script Complete ---")