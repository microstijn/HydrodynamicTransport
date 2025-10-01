# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!

using ..HydrodynamicTransport.ModelStructs
using LinearAlgebra

# ... (solve_implicit_diffusion_column! is unchanged) ...
function solve_implicit_diffusion_column!(
    C_out_col::AbstractVector,
    C_in_col::AbstractVector,
    grid::CartesianGrid,
    i::Int, j::Int,
    dt::Float64, Kz::Float64
)
    nz = length(C_in_col)
    dz = [grid.volume[i,j,k] / grid.face_area_z[i,j,k] for k in 1:nz]
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


function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kz = 1e-4
    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return; end

    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        C_in = deepcopy(C)
        C_after_advection = similar(C)

        # --- FIX: Loop over physical domain and use global indices ---
        for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            C_col_in = C_in[i_glob, j_glob, :]
            C_col_out = view(C_after_advection, i_glob, j_glob, :)
            
            flux_z = zeros(nz + 1)
            for k in 2:nz
                velocity = state.w[i_glob, j_glob, k]
                concentration_at_face = velocity >= 0 ? C_col_in[k-1] : C_col_in[k]
                face_area = isa(grid, CartesianGrid) ? grid.face_area_z[i_glob,j_glob,k] : 1/(grid.pm[i_phys,j_phys]*grid.pn[i_phys,j_phys])
                flux_z[k] = velocity * concentration_at_face * face_area
            end

            for k in 1:nz
                flux_divergence = flux_z[k+1] - flux_z[k]
                volume = grid.volume[i_glob,j_glob,k]
                if volume > 0
                    C_col_out[k] = C_col_in[k] - (dt / volume) * flux_divergence
                else
                    C_col_out[k] = C_col_in[k]
                end
            end
        end

        # --- FIX: Loop over physical domain and use global indices ---
        for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            C_col_in = C_after_advection[i_glob, j_glob, :]
            C_col_out = view(C, i_glob, j_glob, :)
            
            if isa(grid, CartesianGrid)
                solve_implicit_diffusion_column!(C_col_out, C_col_in, grid, i_glob, j_glob, dt, Kz)
            end
        end
    end
    return nothing
end

end # module VerticalTransportModule
