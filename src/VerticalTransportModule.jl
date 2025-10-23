# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!, advect_diffuse_implicit_z!

using ..HydrodynamicTransport.ModelStructs
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
                
                solve_implicit_diffusion_column!(C_out_col, C_col_in, grid, i_glob, j_glob, dt, Kz)
            end
        end
    end
    return nothing
end

# ==============================================================================
# --- NEW: 3D Implicit Advection-Diffusion (Crank-Nicolson ADI) ---
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


end # module VerticalTransportModule

