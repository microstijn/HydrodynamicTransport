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

# In src/VerticalTransportModule.jl

function advect_diffuse_tvd_implicit_z!(
    C_out::Array{Float64, 3},
    C_in::Array{Float64, 3},
    state::State,
    grid::AbstractGrid,
    dt::Float64,
    Kz::Float64,
    limiter_func::Function,
    D_crit::Float64
)
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    w_hydro = state.w
    volume_threshold = 1e-12 # Threshold for considering volume zero
    if nz <= 1; C_out .= C_in; return; end

    flux_f_fou = Vector{Float64}(undef, nz + 1)
    flux_f_lim = Vector{Float64}(undef, nz + 1)

    Threads.@threads for j_phys in 1:ny

        a = Vector{Float64}(undef, nz - 1)
        b = Vector{Float64}(undef, nz)
        c = Vector{Float64}(undef, nz - 1)
        d = Vector{Float64}(undef, nz)

        for i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            # --- Step 1: Calculate Advection Fluxes (Includes D_crit checks) ---
            # ... (This part remains unchanged from the previous version) ...
            flux_f_fou[1] = 0.0; flux_f_lim[1] = 0.0
            flux_f_fou[nz+1] = 0.0; flux_f_lim[nz+1] = 0.0
            for k_phys_face in 2:nz
                velocity = w_hydro[i_glob, j_glob, k_phys_face]
                local upstream_k
                if velocity >= 0; upstream_k = k_phys_face - 1; else; upstream_k = k_phys_face; end
                if 1 <= upstream_k <= nz
                    dz_upstream = get_dz_centers(grid, i_glob, j_glob, upstream_k)
                    if dz_upstream < D_crit; velocity = 0.0; end
                else; velocity = 0.0; end
                if abs(velocity) < 1e-12
                    flux_f_fou[k_phys_face] = 0.0; flux_f_lim[k_phys_face] = 0.0; continue
                end
                local donor_idx, receiver_idx, c_up_far, c_up_near, c_down_near
                if velocity >= 0; donor_idx = k_phys_face - 1; receiver_idx = k_phys_face;
                else; donor_idx = k_phys_face; receiver_idx = k_phys_face - 1; end
                c_up_near = C_in[i_glob, j_glob, donor_idx]; c_down_near = C_in[i_glob, j_glob, receiver_idx]
                face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, k_phys_face] : 1.0 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                flux_f_fou[k_phys_face] = velocity * c_up_near * face_area
                if (velocity >= 0 && donor_idx <= 1) || (velocity < 0 && donor_idx >= nz-1) # Typo fixed: nz-1, not nz
                    flux_f_lim[k_phys_face] = flux_f_fou[k_phys_face]
                else
                    if velocity >= 0; c_up_far = C_in[i_glob, j_glob, donor_idx - 1];
                    else; c_up_far = C_in[i_glob, j_glob, donor_idx + 1]; end
                    flux_f_lim[k_phys_face] = calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area, limiter_func)
                end
            end


            # --- Step 2: Build and Solve System ---
            for k_phys in 1:nz
                 cell_volume = grid.volume[i_glob, j_glob, k_phys]

                # --- FIX: Handle Dry Cells ---
                if cell_volume < volume_threshold
                    if k_phys > 1;  a[k_phys - 1] = 0.0; end
                    b[k_phys] = 1.0
                    if k_phys < nz; c[k_phys]     = 0.0; end
                    d[k_phys] = C_in[i_glob, j_glob, k_phys] # Ensure C_out = C_in
                    continue # Skip rest of calculation
                end
                # --- END FIX ---

                # --- Calculations for WET cells ---
                w_bottom = w_hydro[i_glob, j_glob, k_phys]
                w_top    = w_hydro[i_glob, j_glob, k_phys + 1]

                 # --- WET/DRY Check for Cr numbers ---
                if w_bottom >= 0 && k_phys > 1 && get_dz_centers(grid, i_glob, j_glob, k_phys-1) < D_crit; w_bottom=0.0; end
                if w_bottom < 0 && get_dz_centers(grid, i_glob, j_glob, k_phys) < D_crit; w_bottom=0.0; end
                if w_top >= 0 && get_dz_centers(grid, i_glob, j_glob, k_phys) < D_crit; w_top=0.0; end
                if w_top < 0 && k_phys < nz && get_dz_centers(grid, i_glob, j_glob, k_phys+1) < D_crit; w_top=0.0; end
                 # ---

                dz_k = get_dz_at_face(grid, i_glob, j_glob, k_phys)
                dz_kp1 = get_dz_at_face(grid, i_glob, j_glob, k_phys + 1)
                cr_bottom = (dz_k > 1e-9) ? (dt / dz_k) * w_bottom : 0.0
                cr_top    = (dz_kp1 > 1e-9) ? (dt / dz_kp1) * w_top : 0.0
                alpha_adv = max(cr_bottom, 0)
                gamma_adv = min(cr_top, 0)
                beta_adv = max(cr_top, 0) - min(cr_bottom, 0)

                dz_centers = get_dz_centers(grid, i_glob, j_glob, k_phys)
                D_num = (dz_centers > 1e-9) ? (0.5 * Kz * dt / (dz_centers^2)) : 0.0

                sub_diag  = -alpha_adv - D_num
                sup_diag  =  gamma_adv - D_num
                main_diag =  1.0 + beta_adv + 2.0*D_num

                if k_phys > 1;  a[k_phys - 1] = sub_diag; end
                b[k_phys] = main_diag
                if k_phys < nz; c[k_phys]     = sup_diag; end

                flux_bottom_corr  = flux_f_lim[k_phys]     - flux_f_fou[k_phys]
                flux_top_corr = flux_f_lim[k_phys + 1] - flux_f_fou[k_phys + 1]
                flux_divergence_corr = flux_top_corr - flux_bottom_corr
                # Volume is guaranteed > volume_threshold here
                RHS_adv_corr = - (dt / cell_volume) * flux_divergence_corr

                C_center = C_in[i_glob, j_glob, k_phys]
                C_bottom = (k_phys == 1)  ? C_center : C_in[i_glob, j_glob, k_phys - 1]
                C_top    = (k_phys == nz) ? C_center : C_in[i_glob, j_glob, k_phys + 1]
                RHS_diff = C_center * (1.0 - 2.0*D_num) + C_bottom * D_num + C_top * D_num

                d[k_phys] = RHS_diff + RHS_adv_corr

            end # end cell loop (k_phys)

            # --- Adjust Matrix for Boundary Conditions & Solve ---
             if grid.volume[i_glob, j_glob, 1] >= volume_threshold && nz > 1 # Check if cell 1 is wet
                 b[1]  += a[1];  a[1] = 0.0
             elseif nz > 1 # Cell 1 is dry
                 b[1] = 1.0
             end
             if grid.volume[i_glob, j_glob, nz] >= volume_threshold && nz > 1 # Check if cell nz is wet
                 b[nz] += c[nz-1]; c[nz-1] = 0.0
             elseif nz > 1 # Cell nz is dry
                 b[nz] = 1.0
             end
             # Handle nz=1 case
             if nz == 1 && grid.volume[i_glob, j_glob, 1] < volume_threshold
                 b[1] = 1.0
             end


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

