# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!

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


# --- Main transport function ---
function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64, sediment_tracers::Dict{Symbol, SedimentParams})
    Kz = 1e-4
    g = 9.81
    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return; end

    for (tracer_name, C_final) in state.tracers
        C_buffer = state._buffers[tracer_name]
        is_sediment = haskey(sediment_tracers, tracer_name)
        
        @inbounds for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            C_col_in = view(C_final, i_glob, j_glob, :)
            C_col_out = view(C_buffer, i_glob, j_glob, :)
            flux_z_col = view(state.flux_z, i_glob, j_glob, :)
            flux_z_col .= 0.0

            # Advection for all cells
            for k in 2:nz
                velocity = state.w[i_glob, j_glob, k]
                concentration_at_face = velocity >= 0 ? C_col_in[k-1] : C_col_in[k]
                face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, k] : 1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                flux_z_col[k] = velocity * concentration_at_face * face_area
            end

            # Update concentrations based on advection
            for k in 1:nz
                flux_divergence = flux_z_col[k+1] - flux_z_col[k]
                volume = grid.volume[i_glob, j_glob, k]
                if volume > 0; C_col_out[k] = C_col_in[k] - (dt / volume) * flux_divergence; else; C_col_out[k] = C_col_in[k]; end
            end

            # If it's a sediment tracer, apply bed flux to the post-advection concentration
            if is_sediment
                params = sediment_tracers[tracer_name]
                C_b_after_advection = max(0.0, C_col_out[1])
                bed_mass_array = state.bed_mass[tracer_name]
                
                bottom_face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, 1] : 1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                dz_bottom = grid.volume[i_glob, j_glob, 1] / bottom_face_area
                
                u_bottom = 0.5 * (state.u[i_glob, j_glob, 1] + state.u[i_glob+1, j_glob, 1])
                v_bottom = 0.5 * (state.v[i_glob, j_glob, 1] + state.v[i_glob, j_glob+1, 1])
                Cd = g * params.manning_n^2 / (dz_bottom^(1/3))
                tau_b = params.rho_fluid * Cd * (u_bottom^2 + v_bottom^2)
                
                Cv = C_b_after_advection / params.rho_particle
                ws_effective = params.ws0 * (1.0 - min(1.0, Cv))^params.n_exponent
                
                Fd = (tau_b < params.tau_d) ? ws_effective * C_b_after_advection * (1 - tau_b / params.tau_d) : 0.0
                Fr = (tau_b > params.tau_cr && bed_mass_array[i_glob, j_glob] > 0.0) ? params.M * (tau_b / params.tau_cr - 1) : 0.0
                
                Fd = min(Fd, (C_b_after_advection * dz_bottom) / dt)
                Fr = min(Fr, bed_mass_array[i_glob, j_glob] / dt)
                
                F_net_bed = Fd - Fr
                bed_mass_array[i_glob, j_glob] += F_net_bed * dt
                
                C_col_out[1] -= (F_net_bed * dt) / dz_bottom
            end
        end

        # Diffusion step for the entire tracer field
        @inbounds for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            C_col_in = C_buffer[i_glob, j_glob, :]
            C_col_out = view(C_final, i_glob, j_glob, :)
            solve_implicit_diffusion_column!(C_col_out, C_col_in, grid, i_glob, j_glob, dt, Kz)
        end
    end
    return nothing
end

end # module VerticalTransportModule