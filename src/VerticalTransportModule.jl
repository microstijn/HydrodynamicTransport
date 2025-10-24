# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!
export advect_diffuse_implicit_z!
export advect_diffuse_tvd_implicit_z!

using ..HydrodynamicTransport.ModelStructs
using ..FluxLimitersModule: calculate_limited_flux
using LinearAlgebra

# --- Diffusion Solver for a single Cartesian column ---
function solve_implicit_diffusion_column!(
    C_out_col::AbstractVector,
    C_in_col::AbstractVector,
    grid::CartesianGrid,
    i_glob::Int, j_glob::Int,
    dt::Float64, Kz::Float64
)
    nz = length(C_in_col)
    if nz <= 1; C_out_col .= C_in_col; return; end
    
    @inbounds dz = [grid.volume[i_glob,j_glob,k] / grid.face_area_z[i_glob,j_glob,k] for k in 1:nz]
    alpha = 0.5 * Kz * dt ./ (dz .* dz)
    
    lower_A = -alpha[2:end]; main_A  = 1.0 .+ 2.0 .* alpha; upper_A = -alpha[1:end-1]
    main_A[1] = 1.0 + 2.0 * alpha[1]; upper_A[1] = -2.0 * alpha[1]
    main_A[end] = 1.0 + 2.0 * alpha[end]; lower_A[end] = -2.0 * alpha[end]
    A = Tridiagonal(lower_A, main_A, upper_A)
    
    lower_B = alpha[2:end]; main_B  = 1.0 .- 2.0 .* alpha; upper_B = alpha[1:end-1]
    main_B[1] = 1.0 - 2.0 * alpha[1]; upper_B[1] = 2.0 * alpha[1]
    main_B[end] = 1.0 - 2.0 * alpha[end]; lower_B[end] = 2.0 * alpha[end]
    B = Tridiagonal(lower_B, main_B, upper_B)
    
    rhs = B * C_in_col
    C_out_col .= A \ rhs
end

# --- Diffusion Solver for a single Curvilinear column ---
function solve_implicit_diffusion_column!(
    C_out_col::AbstractVector,
    C_in_col::AbstractVector,
    grid::CurvilinearGrid,
    i_glob::Int, j_glob::Int,
    dt::Float64, Kz::Float64
)
    nz = length(C_in_col)
    if nz <= 1; C_out_col .= C_in_col; return; end

    @inbounds dz_vec = abs.(grid.z_w[2:end] - grid.z_w[1:end-1])
    alpha = 0.5 * Kz * dt ./ (dz_vec .* dz_vec)
    
    lower_A = -alpha[2:end]; main_A  = 1.0 .+ 2.0 .* alpha; upper_A = -alpha[1:end-1]
    main_A[1] = 1.0 + 2.0 * alpha[1]; upper_A[1] = -2.0 * alpha[1]
    main_A[end] = 1.0 + 2.0 * alpha[end]; lower_A[end] = -2.0 * alpha[end]
    A = Tridiagonal(lower_A, main_A, upper_A)
    
    lower_B = alpha[2:end]; main_B  = 1.0 .- 2.0 .* alpha; upper_B = alpha[1:end-1]
    main_B[1] = 1.0 - 2.0 * alpha[1]; upper_B[1] = 2.0 * alpha[1]
    main_B[end] = 1.0 - 2.0 * alpha[end]; lower_B[end] = 2.0 * alpha[end]
    B = Tridiagonal(lower_B, main_B, upper_B)
    
    rhs = B * C_in_col
    C_out_col .= A \ rhs
end


# --- Main transport function (Multithreaded) ---
function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kz = 1e-4
    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return; end

    for tracer_name in keys(state.tracers)
        C_final = state.tracers[tracer_name]
        C_buffer = state._buffer1[tracer_name]

        # --- 1. Advection Step ---
        # This loop is parallelized. Each thread handles a different set of water columns.
        Threads.@threads for j_phys in 1:ny
            @inbounds for i_phys in 1:nx
                i_glob, j_glob = i_phys + ng, j_phys + ng

                C_col_in = C_final[i_glob, j_glob, :]
                C_col_out = view(C_buffer, i_glob, j_glob, :)
                
                flux_z_col = view(state.flux_z, i_glob, j_glob, :)
                flux_z_col .= 0.0

                for k in 2:nz
                    velocity = state.w[i_glob, j_glob, k]
                    concentration_at_face = velocity >= 0 ? C_col_in[k-1] : C_col_in[k]
                    
                    face_area = if isa(grid, CartesianGrid)
                        grid.face_area_z[i_glob, j_glob, k]
                    else # CurvilinearGrid
                        1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                    end
                    flux_z_col[k] = velocity * concentration_at_face * face_area
                end

                for k in 1:nz
                    flux_divergence = flux_z_col[k+1] - flux_z_col[k]
                    volume = grid.volume[i_glob, j_glob, k]
                    if volume > 0
                        C_col_out[k] = C_col_in[k] - (dt / volume) * flux_divergence
                    else
                        C_col_out[k] = C_col_in[k]
                    end
                end
            end
        end

        # --- 2. Diffusion Step ---
        # This loop is also parallelized, as each column's implicit solve is independent.
        Threads.@threads for j_phys in 1:ny
            @inbounds for i_phys in 1:nx
                i_glob, j_glob = i_phys + ng, j_phys + ng

                C_col_in = C_buffer[i_glob, j_glob, :]
                C_col_out = view(C_final, i_glob, j_glob, :)
                
                solve_implicit_diffusion_column!(C_col_out, C_col_in, grid, i_glob, j_glob, dt, Kz)
            end
        end
    end
    return nothing
end

# ==============================================================================
# 3D Implicit Advection-Diffusion (Crank-Nicolson ADI) 
# ==============================================================================

function advect_diffuse_implicit_z!(C_out::Array{Float64, 3}, C_in::Array{Float64, 3}, state::State, grid::AbstractGrid, dt::Float64, Kz::Float64)
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    w = state.w
    if nz <= 1; C_out .= C_in; return; end

    a_threads = [Vector{Float64}(undef, nz - 1) for _ in 1:Threads.nthreads()]
    b_threads = [Vector{Float64}(undef, nz)     for _ in 1:Threads.nthreads()]
    c_threads = [Vector{Float64}(undef, nz - 1) for _ in 1:Threads.nthreads()]
    d_threads = [Vector{Float64}(undef, nz)     for _ in 1:Threads.nthreads()]

    Threads.@threads for j_phys in 1:ny
        tid = Threads.threadid()
        a, b, c, d = a_threads[tid], b_threads[tid], c_threads[tid], d_threads[tid]

        for i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            for k_phys in 1:nz
                dz = isa(grid, CartesianGrid) ? grid.volume[i_glob,j_glob,k_phys] / grid.face_area_z[i_glob,j_glob,k_phys] : abs(grid.z_w[k_phys+1] - grid.z_w[k_phys])
                
                w_bottom = w[i_glob, j_glob, k_phys]
                w_top    = w[i_glob, j_glob, k_phys + 1]

                Cr_bottom = 0.5 * w_bottom * dt / dz
                Cr_top    = 0.5 * w_top    * dt / dz
                D_num     = 0.5 * Kz * dt / (dz^2)

                sub_diag  = -Cr_bottom - D_num
                sup_diag  =  Cr_top    - D_num
                
                local main_diag
                if k_phys == 1 # Bottom boundary
                    main_diag = 1.0 + Cr_top - Cr_bottom + D_num
                elseif k_phys == nz # Top boundary
                    main_diag = 1.0 + Cr_top - Cr_bottom + D_num
                else # Interior cells
                    main_diag = 1.0 + Cr_top - Cr_bottom + 2.0*D_num
                end

                if k_phys > 1;  a[k_phys - 1] = sub_diag; end
                if k_phys < nz; c[k_phys]     = sup_diag; end
                b[k_phys] = main_diag

                # --- CORRECTED: RHS vector calculation with boundary logic ---
                if k_phys == 1 # Bottom boundary
                     d[k_phys] = C_in[i_glob, j_glob, k_phys] * (1.0 - (Cr_top - Cr_bottom) - D_num) +
                                 C_in[i_glob, j_glob, k_phys+1] * (-Cr_top + D_num)
                elseif k_phys == nz # Top boundary
                     d[k_phys] = C_in[i_glob, j_glob, k_phys] * (1.0 - (Cr_top - Cr_bottom) - D_num) +
                                 C_in[i_glob, j_glob, k_phys-1] * (Cr_bottom + D_num)
                else # Interior cells
                    C_in_bottom = C_in[i_glob, j_glob, k_phys-1]
                    C_in_top = C_in[i_glob, j_glob, k_phys+1]
                    d[k_phys] = C_in[i_glob, j_glob, k_phys] * (1.0 - (Cr_top - Cr_bottom) - 2.0*D_num) +
                                C_in_bottom * (Cr_bottom + D_num) +
                                C_in_top    * (-Cr_top + D_num)
                end
            end
            
            # --- REMOVED: Erroneous adjustment block ---
            
            A = Tridiagonal(a, b, c)
            solution = A \ d
            view(C_out, i_glob, j_glob, :) .= solution
        end
    end
end

# ==============================================================================
# 3D Implicit Advection-Diffusion (TVD) 
# ==============================================================================

function advect_diffuse_tvd_implicit_z!(C_out::Array{Float64, 3}, C_in::Array{Float64, 3}, state::State, grid::AbstractGrid, dt::Float64, Kz::Float64, limiter_func::Function)
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    w = state.w
    if nz <= 1; C_out .= C_in; return; end

    # Pre-allocate buffers for the corrective fluxes (one per column)
    flux_f_fou = Vector{Float64}(undef, nz + 1)
    flux_f_lim = Vector{Float64}(undef, nz + 1)

    # Threading over the horizontal plane
    Threads.@threads for j_phys in 1:ny
        
        # Thread-local buffers for the tridiagonal system
        a = Vector{Float64}(undef, nz - 1) # sub-diagonal
        b = Vector{Float64}(undef, nz)     # main diagonal
        c = Vector{Float64}(undef, nz - 1) # super-diagonal
        d = Vector{Float64}(undef, nz)     # RHS

        for i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            # --- Step 1: Calculate Advection Fluxes (TVD and FOU) ---
            
            # --- Boundary Faces (k=1 and k=nz+1) ---
            # Enforce zero flux at the solid bottom and top boundaries
            flux_f_fou[1] = 0.0
            flux_f_lim[1] = 0.0
            flux_f_fou[nz+1] = 0.0
            flux_f_lim[nz+1] = 0.0

            # --- Interior Faces (k=2 to k=nz) ---
            for k_phys_face in 2:nz
                
                velocity = w[i_glob, j_glob, k_phys_face]
                local c_up_far, c_up_near, c_down_near
                
                if abs(velocity) < 1e-12
                    flux_f_fou[k_phys_face] = 0.0
                    flux_f_lim[k_phys_face] = 0.0
                    continue
                end

                local donor_idx, receiver_idx
                if velocity >= 0 # Flow Bottom->Top (positive k)
                    donor_idx    = k_phys_face - 1
                    receiver_idx = k_phys_face
                else # Flow Top->Bottom (negative k)
                    donor_idx    = k_phys_face
                    receiver_idx = k_phys_face - 1
                end
                
                c_up_near    = C_in[i_glob, j_glob, donor_idx]
                c_down_near  = C_in[i_glob, j_glob, receiver_idx]

                # Get face area
                face_area = if isa(grid, CartesianGrid)
                    # Use the face area of the *donor* cell
                    grid.face_area_z[i_glob, j_glob, donor_idx]
                else
                    1.0 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                end
                
                # a) Low-order First-Order Upwind (FOU) flux
                flux_f_fou[k_phys_face] = velocity * c_up_near * face_area
                
                # --- b) High-order limited flux (TVD) ---
                # Use low-order FOU at boundary-adjacent faces where we can't get c_up_far
                if (velocity >= 0 && k_phys_face == 2) || (velocity < 0 && k_phys_face == nz)
                    flux_f_lim[k_phys_face] = flux_f_fou[k_phys_face]
                else
                    # This is now safe, as k-1 and k+1 are valid
                    if velocity >= 0
                        c_up_far = C_in[i_glob, j_glob, donor_idx - 1]
                    else
                        c_up_far = C_in[i_glob, j_glob, donor_idx + 1]
                    end
                    flux_f_lim[k_phys_face] = calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area, limiter_func)
                end
            end # end face loop

            # --- Step 2: Build and Solve the Tridiagonal System (Cell Loop) ---
            for k_phys in 1:nz
                
                # --- Advection Terms (FOU) ---
                w_bottom = w[i_glob, j_glob, k_phys]
                w_top    = w[i_glob, j_glob, k_phys + 1]
                
                # --- FIX: Call the new, robust helper ---
                dz_k = get_dz_at_face(grid, i_glob, j_glob, k_phys)
                dz_kp1 = get_dz_at_face(grid, i_glob, j_glob, k_phys + 1)
                
                cr_bottom = (dt / dz_k) * w_bottom
                cr_top    = (dt / dz_kp1) * w_top
            
                alpha_adv = max(cr_bottom, 0)
                gamma_adv = min(cr_top, 0)
                beta_adv = max(cr_top, 0) - min(cr_bottom, 0)
                
                # --- Diffusion Terms (Crank-Nicolson) ---
                dz_centers = get_dz_centers(grid, i_glob, j_glob, k_phys)
                D_num = 0.5 * Kz * dt / (dz_centers^2)
                
                # --- LHS: Implicit FOU Advection + Implicit CN Diffusion ---
                sub_diag  = -alpha_adv - D_num
                sup_diag  =  gamma_adv - D_num
                main_diag =  1.0 + beta_adv + 2.0*D_num
                
                if k_phys > 1;  a[k_phys - 1] = sub_diag; end
                b[k_phys] = main_diag
                if k_phys < nz; c[k_phys]     = sup_diag; end
                
                # --- RHS: Explicit Advection Correction + Explicit CN Diffusion ---
                
                # a) Advection Correction
                flux_bottom_corr  = flux_f_lim[k_phys]     - flux_f_fou[k_phys]
                flux_top_corr = flux_f_lim[k_phys + 1] - flux_f_fou[k_phys + 1]
                flux_divergence_corr = flux_top_corr - flux_bottom_corr
                RHS_adv_corr = - (dt / grid.volume[i_glob, j_glob, k_phys]) * flux_divergence_corr

                # b) Explicit CN Diffusion (with boundary conditions)
                # Apply no-flux (zero-gradient) condition: C_bottom = C_center, C_top = C_center
                C_center = C_in[i_glob, j_glob, k_phys]
                C_bottom = (k_phys == 1)  ? C_center : C_in[i_glob, j_glob, k_phys - 1]
                C_top    = (k_phys == nz) ? C_center : C_in[i_glob, j_glob, k_phys + 1]
                
                RHS_diff = C_center * (1.0 - 2.0*D_num) + C_bottom * D_num + C_top * D_num
                
                d[k_phys] = RHS_diff + RHS_adv_corr

            end # end cell loop

            # --- Apply no-flux (zero-gradient) boundary conditions to implicit matrix ---
            # This handles the diffusion part of the LHS
            if nz > 1
                b[1]  += a[1];  a[1] = 0.0
                b[nz] += c[nz-1]; c[nz-1] = 0.0
            end

            # --- Solve the system ---
            A = Tridiagonal(a, b, c)
            solution = A \ d
            view(C_out, i_glob, j_glob, :) .= solution

        end # end i_phys loop
    end # end j_phys loop
end

# helper function for dz centers (needed for flux limiting)

@inline get_dz_centers(grid::CartesianGrid, i_glob, j_glob, k_cell) = grid.volume[i_glob,j_glob,k_cell] / grid.face_area_z[i_glob,j_glob,k_cell]
@inline get_dz_centers(grid::CurvilinearGrid, i_glob, j_glob, k_cell) = abs(grid.z_w[k_cell+1] - grid.z_w[k_cell])

"""
    get_dz_at_face(grid, i_glob, j_glob, k_face)

Calculates the distance between cell centers for an interior face,
or the distance from a cell center to the boundary for a boundary face.
"""
@inline function get_dz_at_face(grid::AbstractGrid, i_glob, j_glob, k_face)
    nz = isa(grid, CartesianGrid) ? grid.dims[3] : grid.nz
    
    if k_face == 1 # Solid bottom boundary face
        # Distance from boundary to center of cell 1
        return 0.5 * get_dz_centers(grid, i_glob, j_glob, 1)
    elseif k_face == nz + 1 # Solid top boundary face
        # Distance from center of cell nz to boundary
        return 0.5 * get_dz_centers(grid, i_glob, j_glob, nz)
    else # Interior face (k_face from 2 to nz)
        # Distance between center of cell k_face-1 and cell k_face
        return 0.5 * (get_dz_centers(grid, i_glob, j_glob, k_face-1) + get_dz_centers(grid, i_glob, j_glob, k_face))
    end
end

end # module VerticalTransportModule

