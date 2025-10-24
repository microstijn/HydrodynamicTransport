# test/test_tvd_stability_curv.jl

using Test
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.HydrodynamicsModule
using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.SettlingModule # Needed for apply_settling!
using HydrodynamicTransport.BedExchangeModule
using HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.FluxLimitersModule
using LinearAlgebra # For Tridiagonal

println("\n--- Running TVD Implicit Stability Test Script (Curvilinear Grid) ---")

# ==============================================================================
# --- 1. Setup Simplified 3D Curvilinear Test Case ---
# ==============================================================================
nx, ny, nz = 10, 10, 5
Lx, Ly, Lz = 100.0, 100.0, 50.0
dx = Lx / nx # 10.0
dy = Ly / ny # 10.0
ng = 2

# --- Manually construct a simple, rectangular CurvilinearGrid ---
nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

pm_full = fill(1.0 / dx, (nx_tot, ny_tot)) # pm = 0.1
pn_full = fill(1.0 / dy, (nx_tot, ny_tot)) # pn = 0.1
angle_full = zeros(nx_tot, ny_tot)
h_full = fill(Lz, (nx_tot, ny_tot))
mask_rho_full = ones(Bool, nx_tot, ny_tot)
mask_u_full = ones(Bool, nx_tot + 1, ny_tot)
mask_v_full = ones(Bool, nx_tot, ny_tot + 1)
z_w_vec = collect(range(-Lz, 0.0, length=nz+1))
dz = abs(z_w_vec[2] - z_w_vec[1]) # dz = 10.0

lon_rho, lat_rho = zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot)
lon_u, lat_u = zeros(nx_tot + 1, ny_tot), zeros(nx_tot + 1, ny_tot)
lon_v, lat_v = zeros(nx_tot, ny_tot + 1), zeros(nx_tot, ny_tot + 1)

face_area_x = fill(dy * dz, (nx_tot + 1, ny_tot, nz))
face_area_y = fill(dx * dz, (nx_tot, ny_tot + 1, nz))
volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))

grid = CurvilinearGrid( # Use the CurvilinearGrid constructor
    ng, nx, ny, nz,
    lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
    z_w_vec,
    pm_full, pn_full, angle_full, h_full,
    mask_rho_full, mask_u_full, mask_v_full,
    face_area_x, face_area_y, volume
)

# --- State Initialization ---
tracer_names = (:C, :Sediment)
sediment_tracer_list = [:Sediment]
state = initialize_state(grid, tracer_names; sediment_tracers=sediment_tracer_list)

# --- Initial Condition: Sharp step in C, uniform Sediment ---
step_loc_x = ng + 5
step_loc_y = ng + 5
state.tracers[:C][ng+1:step_loc_x, ng+1:step_loc_y, :] .= 1.0
state.tracers[:C][step_loc_x+1:nx+ng, :, :] .= 0.0
state.tracers[:C][:, step_loc_y+1:ny+ng, :] .= 0.0
state.tracers[:Sediment] .= 0.5

println("Initial Min/Max C: ", minimum(state.tracers[:C]), "/", maximum(state.tracers[:C]))
println("Initial Min/Max Sediment: ", minimum(state.tracers[:Sediment]), "/", maximum(state.tracers[:Sediment]))

# --- Hydrodynamics: Constant diagonal flow (strongest near bottom) ---
u_val, v_val, w_val = 2.0, 2.0, 0.5
state.u[:, :, 1] .= u_val # Strongest u near bottom
state.v[:, :, 1] .= v_val # Strongest v near bottom
state.w .= w_val         # Uniform w

# --- Parameters ---
start_time = 0.0
# Advect ~5 cells using grid metrics. dx = 1/pm = 10. Time = distance/velocity = 5*10 / 2.0 = 25.0
end_time = 25.0
dt = 1.0 # Initial dt
use_adaptive_dt = true
cfl_max = 0.8
dt_max = 10.0
dt_min = 0.01
dt_growth_factor = 1.1
Kh = 0.0 # No horizontal diffusion
Kz = 0.0 # No vertical diffusion
limiter_func = van_leer
advection_scheme = :ImplicitADI_3D
sediment_params = Dict(:Sediment => SedimentParams(ws=0.1, erosion_rate = 0.0))
boundary_conditions = [OpenBoundary(side=:West), OpenBoundary(side=:East), OpenBoundary(side=:North), OpenBoundary(side=:South)]
sources = PointSource[]
functional_interactions = FunctionalInteraction[]
D_crit = 0.0

# ==============================================================================
# --- 2. Replicate Time Stepping Loop with Checks ---
# ==============================================================================
time = start_time
current_dt = dt
step_count = 0

min_c_overall = Inf
min_sed_overall = Inf

while time < end_time && step_count < 50 # Increase step limit slightly
    step_count += 1
    println("\n--- Step: $step_count, Current Time: $(round(time, digits=2)) ---")

    trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt
    trial_dt = min(trial_dt, end_time - time)
    if use_adaptive_dt && trial_dt < dt_min; println("Timestep too small, stopping."); break; end
    if trial_dt < 1e-9; break; end

    println("Attempting trial_dt = $(round(trial_dt, digits=3))")

    step_successful = false
    while !step_successful
        state_backup = deepcopy(state)

        function check_min(label::String, state_to_check::State)
            min_c = minimum(state_to_check.tracers[:C])
            min_s = minimum(state_to_check.tracers[:Sediment])
            is_negative = min_c < -1e-9 || min_s < -1e-9
            println("  - Min C/Sed after $label: $(round(min_c, digits=4)) / $(round(min_s, digits=4)) $(is_negative ? "!!!" : "")")
            return is_negative
        end

        # --- Hydrodynamics & Boundaries ---
        update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
        apply_boundary_conditions!(state_backup, grid, boundary_conditions)
        if check_min("Boundaries", state_backup); print("Negative after Boundaries"); end

        local deposition

        # --- Transport Step (:ImplicitADI_3D only) ---
        if advection_scheme != :ImplicitADI_3D
            error("This script only tests :ImplicitADI_3D")
        end

        for (tracer_name, C_initial) in state_backup.tracers
            C_buffer1 = state_backup._buffer1[tracer_name]
            C_buffer2 = state_backup._buffer2[tracer_name]

            advect_diffuse_tvd_implicit_x!(C_buffer1, C_initial, state_backup, grid, trial_dt, Kh, limiter_func)
            if check_min("X-Sweep ($tracer_name)", state_backup); print("Negative after X-Sweep ($tracer_name)"); end

            advect_diffuse_tvd_implicit_y!(C_buffer2, C_buffer1, state_backup, grid, trial_dt, Kh, limiter_func)
            if check_min("Y-Sweep ($tracer_name)", state_backup); print("Negative after Y-Sweep ($tracer_name)"); end

            # Call 7-argument version (no settling here)
            advect_diffuse_tvd_implicit_z!(C_initial, C_buffer2, state_backup, grid, trial_dt, Kz, limiter_func)
            if check_min("Z-Sweep ($tracer_name)", state_backup); print("Negative after Z-Sweep ($tracer_name)"); end
        end

        # --- Settling Step (Separate) ---
        # Note: apply_settling! needs the limiter_func if you upgrade it later
        deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
        if check_min("Settling", state_backup); print("Negative after Settling"); end

        # --- Physics Steps ---
        bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
        if check_min("Bed Exchange", state_backup); print("Negative after Bed Exchange"); end

        source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
        if check_min("Sources/Sinks", state_backup); print("Negative after Sources/Sinks"); end

        # --- Timestep Validation ---
        cfl_inv_term = calculate_max_cfl_term(state_backup, grid) # Uses the corrected function
        cfl_actual = cfl_inv_term * trial_dt

        println("  - Calculated CFL_actual = $(round(cfl_actual, digits=3)) (Max term = $(round(cfl_inv_term, digits=3)))")

        final_min_c = minimum(state_backup.tracers[:C])
        final_min_s = minimum(state_backup.tracers[:Sediment])
        println("  - Min C/Sed after full step attempt: $(round(final_min_c, digits=4)) / $(round(final_min_s, digits=4))")

        if use_adaptive_dt && cfl_actual > cfl_max
            println("  >>> CFL too high ($cfl_actual > $cfl_max). Reducing dt.")
            trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / (cfl_actual + 1e-9))
            println("      New trial_dt = $(round(trial_dt, digits=3))")
        else
            println("  >>> Step SUCCESSFUL.")
            state = state_backup
            step_successful = true
            if use_adaptive_dt && cfl_actual < 0.5 * cfl_max
                current_dt = min(dt_max, trial_dt * dt_growth_factor)
            else
                current_dt = trial_dt
            end
        end
    end # end while !step_successful

    time += trial_dt
    state.time = time
    min_c_overall = min(min_c_overall, minimum(state.tracers[:C]))
    min_sed_overall = min(min_sed_overall, minimum(state.tracers[:Sediment]))

end # end while time < end_time

# ==============================================================================
# --- 3. Final Check ---
# ==============================================================================
println("\n--- Simulation Finished ---")
println("Final Time: $(round(time, digits=2))")
println("Overall Minimum C encountered: $(round(min_c_overall, digits=4))")
println("Overall Minimum Sediment encountered: $(round(min_sed_overall, digits=4))")

@testset "Final Stability Check (Curvilinear)" begin
    @test min_c_overall >= -1e-9
    @test min_sed_overall >= -1e-9
end

println("\n--- Curvilinear Test Script Complete ---")