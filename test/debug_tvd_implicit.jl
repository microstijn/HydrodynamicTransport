# debug_tvd_implicit.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport

# Assuming your package is activated, otherwise Pkg.activate(...)

using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.HorizontalTransportModule: get_dx_at_face, get_dx_centers
using HydrodynamicTransport.FluxLimitersModule
using LinearAlgebra # For Tridiagonal

println("--- Starting TVD Implicit Debug Script ---")

# ==============================================================================
# --- 1. Setup Minimal 1D Test Case ---
# ==============================================================================
nx, ny, nz = 20, 1, 1
Lx, Ly, Lz = 20.0, 1.0, 1.0
grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
state = initialize_state(grid, (:C,))

C_in = state.tracers[:C]
C_out = state._buffer1[:C] # We'll write the result here
ng = grid.ng

# --- Initial Condition: Sharp Step Change ---
step_location = ng + 10 # Physical index i=10
C_in[ng+1:step_location, ng+1, 1] .= 1.0
C_in[step_location+1:nx+ng, ng+1, 1] .= 0.0

println("Initial Concentration Profile (Physical Cells 1:$nx):")
println(round.(C_in[ng+1:nx+ng, ng+1, 1]', digits=2))

# --- Velocity ---
u_vel = 1.0
state.u .= u_vel

# --- Parameters ---
Kh = 0.0 # Pure Advection
dx = Lx / nx # dx = 1.0
CFL = 0.8
dt = CFL * dx / u_vel # dt = 0.8
limiter_func = van_leer

println("\nParameters:")
println("dx = $dx, dt = $dt, CFL = $CFL")
println("Kh = $Kh")
println("Limiter = $limiter_func")

# ==============================================================================
# --- 2. Isolate and Run ONE Step of advect_diffuse_tvd_implicit_x! ---
# ==============================================================================
println("\n--- Running ONE step of advect_diffuse_tvd_implicit_x! ---")

# Copy the core logic here for detailed printing
# (Alternatively, modify the original function to add prints if preferred)

# Extract necessary variables
u = state.u
k = 1 # Only one layer
j_phys = 1
j_glob = j_phys + ng

# Buffers
flux_f_fou = Vector{Float64}(undef, nx + 1)
flux_f_lim = Vector{Float64}(undef, nx + 1)
a = Vector{Float64}(undef, nx - 1)
b = Vector{Float64}(undef, nx)
c = Vector{Float64}(undef, nx - 1)
d = Vector{Float64}(undef, nx)

# --- Step 1: Calculate Advection Fluxes (TVD and FOU) ---
println("\nCalculating Fluxes at Faces:")
for i_phys_face in 1:(nx + 1)
    i_glob_face = i_phys_face + ng
    
    velocity = u[i_glob_face, j_glob, k]
    local c_up_far, c_up_near, c_down_near
    
    if abs(velocity) < 1e-12
        flux_f_fou[i_phys_face] = 0.0
        flux_f_lim[i_phys_face] = 0.0
        continue
    end

    local donor_idx, receiver_idx
    if velocity >= 0 # Flow L->R
        donor_idx    = i_glob_face - 1
        receiver_idx = i_glob_face
        # Boundary checks needed for c_up_far
        c_up_near    = C_in[donor_idx,    j_glob, k]
        c_down_near  = C_in[receiver_idx, j_glob, k]
        c_up_far     = C_in[max(ng+1, donor_idx - 1), j_glob, k] # Clamp index
    else # Flow R->L (Not used in this test, but keep for completeness)
        donor_idx    = i_glob_face
        receiver_idx = i_glob_face - 1
        # Boundary checks needed for c_up_far
        c_up_near    = C_in[donor_idx,    j_glob, k]
        c_down_near  = C_in[receiver_idx, j_glob, k]
        c_up_far     = C_in[min(nx+ng, donor_idx + 1), j_glob, k] # Clamp index
    end

    face_area = grid.face_area_x[i_glob_face, j_glob, k]
    
    # a) Low-order First-Order Upwind (FOU) flux
    flux_f_fou[i_phys_face] = velocity * c_up_near * face_area

    # b) High-order limited flux (TVD)
    phi = calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area, limiter_func)
    flux_f_lim[i_phys_face] = phi # calculate_limited_flux returns the full flux

    # --- Debug Printing around the step interface ---
    # Physical face indices are 1 to nx+1. Interface is around face 11 (between cell 10 and 11).
    if i_phys_face >= 9 && i_phys_face <= 13
        println("  Face i_phys=$i_phys_face (i_glob=$i_glob_face):")
        println("    Vel=$velocity")
        println("    c_up_far=$(round(c_up_far, digits=3)), c_up_near=$(round(c_up_near, digits=3)), c_down_near=$(round(c_down_near, digits=3))")
        println("    Flux_FOU = $(round(flux_f_fou[i_phys_face], digits=3))")
        println("    Flux_LIM = $(round(flux_f_lim[i_phys_face], digits=3))")
    end
end

# --- Step 2: Build and Solve the Tridiagonal System (Cell Loop) ---
println("\nBuilding Tridiagonal System:")
for i_phys in 1:nx
    i_glob = i_phys + ng
    
    # Advection Terms (FOU)
    u_left = u[i_glob, j_glob, k]
    u_right = u[i_glob + 1, j_glob, k]
    dx_i = get_dx_at_face(grid, i_glob, j_glob)
    dx_ip1 = get_dx_at_face(grid, i_glob + 1, j_glob)
    
    cr_left = (dt / dx_i) * u_left
    cr_right = (dt / dx_ip1) * u_right

    alpha_adv = max(cr_left, 0)
    gamma_adv = min(cr_right, 0)
    beta_adv = max(cr_right, 0) - min(cr_left, 0)
    
    # Diffusion Terms (Crank-Nicolson) - Kh=0
    dx_centers = get_dx_centers(grid, i_glob, j_glob)
    D_num = 0.5 * Kh * dt / (dx_centers^2) # Should be 0.0
    
    # LHS
    sub_diag  = -alpha_adv - D_num
    sup_diag  =  gamma_adv - D_num
    main_diag =  1.0 + beta_adv + 2.0*D_num
    
    if i_phys > 1; a[i_phys - 1] = sub_diag; end
    b[i_phys] = main_diag
    if i_phys < nx; c[i_phys] = sup_diag; end
    
    # RHS
    flux_left_corr  = flux_f_lim[i_phys]     - flux_f_fou[i_phys]
    flux_right_corr = flux_f_lim[i_phys + 1] - flux_f_fou[i_phys + 1]
    flux_divergence_corr = flux_right_corr - flux_left_corr
    RHS_adv_corr = - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence_corr

    C_left   = C_in[max(ng+1, i_glob - 1), j_glob, k] # Clamp index
    C_center = C_in[i_glob,              j_glob, k]
    C_right  = C_in[min(nx+ng, i_glob + 1), j_glob, k] # Clamp index
    RHS_diff = C_center * (1.0 - 2.0*D_num) + C_left * D_num + C_right * D_num # Should just be C_center
    
    d[i_phys] = RHS_diff + RHS_adv_corr

    # --- Debug Printing around the step interface ---
    # Physical cell indices are 1 to nx. Interface is around cell 10 & 11.
    if i_phys >= 9 && i_phys <= 12
        println("  Cell i_phys=$i_phys (i_glob=$i_glob):")
        println("    LHS: a=$(round(sub_diag, digits=3)), b=$(round(main_diag, digits=3)), c=$(round(sup_diag, digits=3))")
        println("    RHS_Adv_Corr = $(round(RHS_adv_corr, digits=3)) (FluxDivCorr=$(round(flux_divergence_corr, digits=3)))")
        println("    RHS_Diff = $(round(RHS_diff, digits=3))")
        println("    RHS_Total (d[$i_phys]) = $(round(d[i_phys], digits=3))")
    end

end

# Apply no-flux boundary conditions to matrix
b[1]  += a[1];  a[1] = 0.0
b[nx] += c[nx-1]; c[nx-1] = 0.0

# Solve
println("\nSolving System...")
A = Tridiagonal(a, b, c)
solution = A \ d
C_out[ng+1:nx+ng, j_glob, k] .= solution

# ==============================================================================
# --- 3. Check Results ---
# ==============================================================================
println("\nFinal Concentration Profile (Physical Cells 1:$nx):")
println(round.(C_out[ng+1:nx+ng, ng+1, 1]', digits=3))

min_C, max_C = minimum(solution), maximum(solution)
println("\nMin concentration in solution: $min_C")
println("Max concentration in solution: $max_C")

if min_C < -1e-9 # Allow for tiny floating point errors
    println("!!! ERROR: Negative concentration detected !!!")
else
    println(">>> Concentrations remained non-negative.")
end