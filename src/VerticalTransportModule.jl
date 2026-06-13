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

    @inbounds dz_vec = [get_dz_centers(grid, i_glob, j_glob, k) for k in 1:nz]   # physical [m], per column
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


# --- Allocation-free per-column Crank-Nicolson vertical diffusion (CurvilinearGrid) ---
# On a sigma grid the layer thickness (hence the CN coefficient alpha) varies per water column,
# so the tridiagonal operator can no longer be factorized once and shared across columns. Instead
# each column is solved directly with the Thomas algorithm using preallocated per-task scratch
# (no allocation, no GC -> retains the bulk of the factorize-once speed-up, which came from
# killing per-column allocation rather than from sharing the LU). Same Crank-Nicolson scheme as
# before (reflective/no-flux top & bottom): the operators A (implicit) and B (explicit) are
# rebuilt from the per-layer `alpha[k] = 0.5·Kz·dt/dz_k²`. For a spatially uniform grid this is
# numerically equivalent to the old shared-operator path (same A, same B).
#
#   alpha  : per-layer CN coefficient (length nz)            [in]
#   dl,dd,du : scratch for A's sub/main/super diagonals      [scratch]
#   rhs    : scratch, receives B·Cin then the solution       [scratch]
#   cprime : Thomas forward-sweep scratch                    [scratch]
@inline function _cn_diffuse_column!(Cout, Cin, alpha, nz, dl, dd, du, rhs, cprime)
    @inbounds begin
        # rhs = B·Cin and assemble A's diagonals. Off-diagonal weight to the single boundary
        # neighbour is doubled (reflective no-flux), matching the original operator.
        for k in 1:nz
            s = (1.0 - 2.0*alpha[k]) * Cin[k]
            if k > 1;  s += (k == nz ? 2.0*alpha[k] : alpha[k]) * Cin[k-1]; end
            if k < nz; s += (k == 1  ? 2.0*alpha[k] : alpha[k]) * Cin[k+1]; end
            rhs[k] = s
            dd[k] = 1.0 + 2.0*alpha[k]
            dl[k] = (k == nz ? -2.0*alpha[k] : -alpha[k])   # sub-diagonal   (row k, couples k-1)
            du[k] = (k == 1  ? -2.0*alpha[k] : -alpha[k])   # super-diagonal (row k, couples k+1)
        end
        # Thomas solve: A·x = rhs, A = Tridiagonal(dl, dd, du).
        cprime[1] = du[1] / dd[1]; rhs[1] = rhs[1] / dd[1]
        for k in 2:nz
            m = dd[k] - dl[k]*cprime[k-1]
            cprime[k] = du[k] / m
            rhs[k] = (rhs[k] - dl[k]*rhs[k-1]) / m
        end
        Cout[nz] = rhs[nz]
        for k in nz-1:-1:1
            Cout[k] = rhs[k] - cprime[k]*Cout[k+1]
        end
    end
    return nothing
end

# --- Main transport function (Multithreaded over tracers) ---
function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kz = 1e-4
    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return; end

    # Tracers are independent -> parallelize over them; each task owns small column scratch.
    tracer_names = collect(keys(state.tracers))
    ntr = length(tracer_names)
    nchunks = max(1, min(Threads.nthreads(), ntr))

    Threads.@threads for cid in 1:nchunks
        flux_col = Vector{Float64}(undef, nz + 1)   # per-task vertical-flux column scratch
        rhs = Vector{Float64}(undef, nz)            # per-task tridiagonal RHS scratch
        # Per-task CN-diffusion column scratch (CurvilinearGrid path).
        alpha = Vector{Float64}(undef, nz); dl = Vector{Float64}(undef, nz)
        dd = Vector{Float64}(undef, nz); du = Vector{Float64}(undef, nz); cprime = Vector{Float64}(undef, nz)
        ti = cid
        while ti <= ntr
            tracer_name = tracer_names[ti]
            C_final = state.tracers[tracer_name]
            C_buffer = state._buffer1[tracer_name]

            # --- 1. Advection Step (serial over columns within this tracer) ---
            @inbounds for j_phys in 1:ny, i_phys in 1:nx
                i_glob, j_glob = i_phys + ng, j_phys + ng
                C_col_in = view(C_final, i_glob, j_glob, :)
                C_col_out = view(C_buffer, i_glob, j_glob, :)

                flux_col .= 0.0
                for k in 2:nz
                    velocity = state.w[i_glob, j_glob, k]
                    concentration_at_face = velocity >= 0 ? C_col_in[k-1] : C_col_in[k]
                    face_area = if isa(grid, CartesianGrid)
                        grid.face_area_z[i_glob, j_glob, k]
                    else # CurvilinearGrid
                        1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                    end
                    flux_col[k] = velocity * concentration_at_face * face_area
                end

                for k in 1:nz
                    flux_divergence = flux_col[k+1] - flux_col[k]
                    volume = grid.volume[i_glob, j_glob, k]
                    if volume > 0
                        C_col_out[k] = C_col_in[k] - (dt / volume) * flux_divergence
                    else
                        C_col_out[k] = C_col_in[k]
                    end
                end
            end

            # --- 2. Diffusion Step (implicit column solve) ---
            if isa(grid, CurvilinearGrid)
                @inbounds for j_phys in 1:ny, i_phys in 1:nx
                    i_glob, j_glob = i_phys + ng, j_phys + ng
                    C_col_in = view(C_buffer, i_glob, j_glob, :)
                    C_col_out = view(C_final, i_glob, j_glob, :)
                    for k in 1:nz
                        dz_k = get_dz_centers(grid, i_glob, j_glob, k)
                        alpha[k] = dz_k > 0.0 ? 0.5 * Kz * dt / (dz_k * dz_k) : 0.0
                    end
                    _cn_diffuse_column!(C_col_out, C_col_in, alpha, nz, dl, dd, du, rhs, cprime)
                end
            else
                @inbounds for j_phys in 1:ny, i_phys in 1:nx
                    i_glob, j_glob = i_phys + ng, j_phys + ng
                    C_col_in = view(C_buffer, i_glob, j_glob, :)
                    C_col_out = view(C_final, i_glob, j_glob, :)
                    solve_implicit_diffusion_column!(C_col_out, C_col_in, grid, i_glob, j_glob, dt, Kz)
                end
            end

            ti += nchunks
        end
    end
    return nothing
end

# ==============================================================================
# 3D Implicit Advection-Diffusion (Crank-Nicolson ADI) 
# ==============================================================================

function advect_diffuse_implicit_z!(C_out::AbstractArray{<:Real, 3}, C_in::AbstractArray{<:Real, 3}, state::State, grid::AbstractGrid, dt::Float64, Kz::Float64)
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
                dz = get_dz_centers(grid, i_glob, j_glob, k_phys)

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

function advect_diffuse_tvd_implicit_z!(C_out::AbstractArray{<:Real, 3}, C_in::AbstractArray{<:Real, 3}, state::State, grid::AbstractGrid, dt::Float64, Kz::Float64, limiter_func::Function)
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
# Physical layer thickness [m] = volume / horizontal cell area = volume·pm·pn. On a sigma grid
# this is Δσ_k·H0(i,j) (spatially varying); on the legacy uniform grid it reduces to |Δz_w|.
@inline get_dz_centers(grid::CurvilinearGrid, i_glob, j_glob, k_cell) = grid.volume[i_glob,j_glob,k_cell] * grid.pm[i_glob,j_glob] * grid.pn[i_glob,j_glob]

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

