# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!, _apply_sedimentation_forward_euler!, _apply_sedimentation_backward_euler!

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
    main_A[1] = 1.0 + alpha[1]; upper_A[1] = -alpha[1]
    main_A[end] = 1.0 + alpha[end]; lower_A[end] = -alpha[end]
    A = Tridiagonal(lower_A, main_A, upper_A)
    
    lower_B = alpha[2:end]; main_B  = 1.0 .- 2.0 .* alpha; upper_B = alpha[1:end-1]
    main_B[1] = 1.0 - alpha[1]; upper_B[1] = alpha[1]
    main_B[end] = 1.0 - alpha[end]; lower_B[end] = alpha[end]
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
    main_A[1] = 1.0 + alpha[1]; upper_A[1] = -alpha[1]
    main_A[end] = 1.0 + alpha[end]; lower_A[end] = -alpha[end]
    A = Tridiagonal(lower_A, main_A, upper_A)
    
    lower_B = alpha[2:end]; main_B  = 1.0 .- 2.0 .* alpha; upper_B = alpha[1:end-1]
    main_B[1] = 1.0 - alpha[1]; upper_B[1] = alpha[1]
    main_B[end] = 1.0 - alpha[end]; lower_B[end] = alpha[end]
    B = Tridiagonal(lower_B, main_B, upper_B)
    
    rhs = B * C_in_col
    C_out_col .= A \ rhs
end

# --- New Generic Implicit Vertical Advection Solver ---
function solve_implicit_vertical_advection_column!(
    C_out_col::AbstractVector,
    C_in_col::AbstractVector,
    velocities::AbstractVector,
    grid::AbstractGrid, i_glob::Int, j_glob::Int,
    dt::Float64
)
    nz = length(C_in_col)
    if nz <= 1; C_out_col .= C_in_col; return; end

    a = Vector{Float64}(undef, nz - 1)
    b = Vector{Float64}(undef, nz)
    c = Vector{Float64}(undef, nz - 1)
    
    face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, 1] : (1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob]))
    if face_area < 1e-12; C_out_col .= C_in_col; return; end

    for k in 1:nz
        V_k = grid.volume[i_glob, j_glob, k]
        if V_k < 1e-9; b[k] = 1.0; if k > 1 a[k-1] = 0.0 end; if k < nz c[k] = 0.0 end; continue; end
        
        cr_k = (velocities[k] * face_area * dt) / V_k
        cr_kp1 = (velocities[k+1] * face_area * dt) / V_k
        
        if k > 1; a[k-1] = -max(cr_k, 0.0); end
        b[k] = 1.0 + max(cr_kp1, 0.0) - min(cr_k, 0.0)
        if k < nz; c[k] = min(cr_kp1, 0.0); end
    end
    
    A = Tridiagonal(a, b, c)
    C_out_col .= A \ C_in_col
end

"""
    _apply_sedimentation_forward_euler!(C_col_out, bed_mass_array, grid, state, params, i_glob, j_glob, dt, g, D_crit)

Calculates and applies the net sediment flux at the bed-water interface for a single water column using the **explicit Forward Euler time-stepping scheme**.
"""
function _apply_sedimentation_forward_euler!(
    C_col_out::AbstractVector,
    bed_mass_array::AbstractMatrix,
    grid::AbstractGrid,
    state::State,
    params::SedimentParams,
    i_glob::Int, j_glob::Int,
    dt::Float64,
    g::Float64,
    D_crit::Float64
)
    if grid.volume[i_glob, j_glob, 1] <= 1.0; return; end
    C_b_after_advection = max(0.0, C_col_out[1])
    bottom_face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, 1] : 1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
    dz_bottom = max(0.01, grid.volume[i_glob, j_glob, 1] / bottom_face_area)
    
    u_bottom = 0.5 * (state.u[i_glob, j_glob, 1] + state.u[i_glob+1, j_glob, 1])
    v_bottom = 0.5 * (state.v[i_glob, j_glob, 1] + state.v[i_glob, j_glob+1, 1])
    Cd = g * params.manning_n^2 / (dz_bottom^(1/3))
    tau_b = params.rho_fluid * Cd * (u_bottom^2 + v_bottom^2)
    
    Cv = C_b_after_advection / params.rho_particle
    ws_effective = params.ws0 * (1.0 - min(1.0, Cv))^params.n_exponent
    
    Fd = (tau_b < params.tau_d) ? ws_effective * C_b_after_advection * (1 - tau_b / params.tau_d) : 0.0
    Fr = (tau_b > params.tau_cr && bed_mass_array[i_glob, j_glob] > 0.0) ? params.M * (tau_b / params.tau_cr - 1) : 0.0
    
    Fd = min(Fd, (C_b_after_advection * dz_bottom) / dt)
    Fr = min(Fr, bed_mass_array[i_glob, j_glob] / (bottom_face_area * dt))
    
    F_net_bed = Fd - Fr
    
    bed_mass_array[i_glob, j_glob] += F_net_bed * bottom_face_area * dt
    C_col_out[1] -= (F_net_bed * dt) / dz_bottom
end


"""
    _apply_sedimentation_backward_euler!(C_col_out, bed_mass_array, grid, state, params, i_glob, j_glob, dt, g, D_crit)

Calculates and applies the net sediment flux at the bed-water interface for a single water column using the **implicit Backward Euler time-stepping scheme**.
"""
function _apply_sedimentation_backward_euler!(
    C_col_out::AbstractVector,
    bed_mass_array::AbstractMatrix,
    grid::AbstractGrid,
    state::State,
    params::SedimentParams,
    i_glob::Int, j_glob::Int,
    dt::Float64,
    g::Float64,
    D_crit::Float64
)
    if grid.volume[i_glob, j_glob, 1] <= 1.0; return; end
    C_b_old = max(0.0, C_col_out[1])
    bottom_face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob, j_glob, 1] : 1 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
    dz_bottom = grid.volume[i_glob, j_glob, 1] / bottom_face_area

    u_bottom = 0.5 * (state.u[i_glob, j_glob, 1] + state.u[i_glob+1, j_glob, 1])
    v_bottom = 0.5 * (state.v[i_glob, j_glob, 1] + state.v[i_glob, j_glob+1, 1])
    Cd = g * params.manning_n^2 / (dz_bottom^(1/3))
    tau_b = params.rho_fluid * Cd * (u_bottom^2 + v_bottom^2)
    
    Fr = (tau_b > params.tau_cr && bed_mass_array[i_glob, j_glob] > 0.0) ? params.M * (tau_b / params.tau_cr - 1) : 0.0
    Fr = min(Fr, bed_mass_array[i_glob, j_glob] / (bottom_face_area * dt))

    Cv = C_b_old / params.rho_particle
    ws_effective = params.ws0 * (1.0 - min(1.0, Cv))^params.n_exponent
    Fd_factor = (tau_b < params.tau_d) ? ws_effective * (1 - tau_b / params.tau_d) : 0.0
    
    numerator = C_b_old + (Fr * dt) / dz_bottom
    denominator = 1.0 + (Fd_factor * dt) / dz_bottom
    C_b_new = numerator / denominator

    C_b_new = max(0.0, C_b_new)
    deposited_mass = (C_b_old - C_b_new) * dz_bottom
    available_mass = C_b_old * dz_bottom + (Fr * dt)
    if deposited_mass > available_mass
         C_b_new = 0.0
    end
    
    Fd_final = Fd_factor * C_b_new
    F_net_bed = Fd_final - Fr
    
    bed_mass_array[i_glob, j_glob] += F_net_bed * bottom_face_area * dt
    C_col_out[1] = C_b_new
end

"""
    vertical_transport!(state, grid, dt, sediment_tracers, D_crit)

Computes the vertical transport of all tracers for a single time step `dt` using an
unconditionally stable, operator-split implicit scheme.
"""
function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64, sediment_tracers::Dict{Symbol, SedimentParams}, D_crit::Float64)
    Kz = 1e-4
    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return; end

    for (tracer_name, C_initial) in state.tracers
        C_advected = state._buffers[tracer_name]
        C_settled = similar(C_initial) # Temporary array for settling step
        is_sediment = haskey(sediment_tracers, tracer_name)
        params = is_sediment ? sediment_tracers[tracer_name] : nothing

        # --- Step 1: Implicit Vertical Advection (due to water velocity w) ---
        @inbounds Base.Threads.@threads for j_phys in 1:ny
            for i_phys in 1:nx
                i_glob, j_glob = i_phys + ng, j_phys + ng
                
                if isa(grid, CurvilinearGrid) && (grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, 1] < D_crit)
                    view(C_advected, i_glob, j_glob, :) .= view(C_initial, i_glob, j_glob, :)
                    continue
                end

                C_in_col = view(C_initial, i_glob, j_glob, :)
                C_out_col = view(C_advected, i_glob, j_glob, :)
                w_faces = view(state.w, i_glob, j_glob, :)
                solve_implicit_vertical_advection_column!(C_out_col, C_in_col, w_faces, grid, i_glob, j_glob, dt)
            end
        end
        C_advected .= max.(0.0, C_advected)

        # --- Step 2: Implicit Vertical Settling (gravitational) ---
        if is_sediment
            ws_velocities = zeros(nz + 1)
            @inbounds Base.Threads.@threads for j_phys in 1:ny
                for i_phys in 1:nx
                    i_glob, j_glob = i_phys + ng, j_phys + ng
                    if isa(grid, CurvilinearGrid) && (grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, 1] < D_crit)
                        view(C_settled, i_glob, j_glob, :) .= view(C_advected, i_glob, j_glob, :)
                        continue
                    end

                    C_in_col = view(C_advected, i_glob, j_glob, :)
                    C_out_col = view(C_settled, i_glob, j_glob, :)
                    
                    for k in 2:nz
                        C_donor = C_in_col[k-1]
                        Cv = C_donor / params.rho_particle
                        ws_effective = params.ws0 * (1.0 - min(1.0, Cv))^params.n_exponent
                        ws_velocities[k] = -ws_effective
                    end
                    solve_implicit_vertical_advection_column!(C_out_col, C_in_col, ws_velocities, grid, i_glob, j_glob, dt)
                end
            end
            C_settled .= max.(0.0, C_settled)
        else
            C_settled .= C_advected
        end
        
        # --- Step 3: Implicit Vertical Diffusion ---
        @inbounds Base.Threads.@threads for j_phys in 1:ny
          for i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            
            if isa(grid, CurvilinearGrid) && (grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, 1] < D_crit)
                view(C_initial, i_glob, j_glob, :) .= view(C_settled, i_glob, j_glob, :)
                continue
            end
            
            C_in_col = view(C_settled, i_glob, j_glob, :)
            C_out_col = view(C_initial, i_glob, j_glob, :) # Write final result back to the main tracer array
            solve_implicit_diffusion_column!(C_out_col, C_in_col, grid, i_glob, j_glob, dt, Kz)
          end
        end
        C_initial .= max.(0.0, C_initial)
    end
    return nothing
end

end # module VerticalTransportModule

