# src/VerticalTransportModule.jl

module VerticalTransportModule

export vertical_transport!

using ..ModelStructs
using LinearAlgebra

# --- Dispatcher function for the correct grid type ---
function vertical_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kz = 1e-4 # m²/s

    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        solve_vertical_column!(C, state.w, grid, dt, Kz)
    end
    return nothing
end

# --- Specialized Methods for each grid type ---
# The logic is nearly identical, but dispatching ensures type stability and future flexibility.

function solve_vertical_column!(C, w, grid::CartesianGrid, dt, Kz)
    nx, ny, nz = grid.dims
    if nz <= 1; return nothing; end

    for j in 1:ny, i in 1:nx
        # Advection Step
        C_advected = C[i, j, :]
        flux_z = zeros(nz + 1)
        for k in 2:nz
            velocity = w[i, j, k]
            concentration_at_face = velocity >= 0 ? C[i, j, k-1] : C[i, j, k]
            flux_z[k] = velocity * concentration_at_face * grid.face_area_z[i, j, k]
        end
        for k in 1:nz
            flux_divergence = flux_z[k+1] - flux_z[k]
            if grid.volume[i, j, k] > 0; C_advected[k] -= (dt / grid.volume[i, j, k]) * flux_divergence; end
        end

        # Diffusion Step
        dz = [grid.volume[i,j,k] / grid.face_area_z[i,j,k] for k in 1:nz]
        alpha = Kz * dt ./ (dz .* dz)
        lower_diag = -alpha[2:end]; main_diag  = 1.0 .+ 2.0 .* alpha; upper_diag = -alpha[1:end-1]
        main_diag[1] = 1.0 + alpha[1]; main_diag[end] = 1.0 + alpha[end]
        A = Tridiagonal(lower_diag, main_diag, upper_diag)
        C[i, j, :] .= A \ C_advected
    end
end

function solve_vertical_column!(C, w, grid::CurvilinearGrid, dt, Kz)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if nz <= 1; return nothing; end

    for j in 1:ny, i in 1:nx
        # Advection Step
        C_advected = C[i, j, :]
        # NOTE: face_area_z is not yet calculated for curvilinear grid, using an approximation.
        face_area_z_approx = 1 / (grid.pm[i,j] * grid.pn[i,j])
        flux_z = zeros(nz + 1)
        for k in 2:nz
            velocity = w[i, j, k]
            concentration_at_face = velocity >= 0 ? C[i, j, k-1] : C[i, j, k]
            flux_z[k] = velocity * concentration_at_face * face_area_z_approx
        end
        for k in 1:nz
            flux_divergence = flux_z[k+1] - flux_z[k]
            if grid.volume[i, j, k] > 0; C_advected[k] -= (dt / grid.volume[i, j, k]) * flux_divergence; end
        end

        # Diffusion Step
        dz_vec = grid.z_w[2:end] - grid.z_w[1:end-1]
        alpha = Kz * dt ./ (dz_vec .* dz_vec)
        lower_diag = -alpha[2:end]; main_diag  = 1.0 .+ 2.0 .* alpha; upper_diag = -alpha[1:end-1]
        main_diag[1] = 1.0 + alpha[1]; main_diag[end] = 1.0 + alpha[end]
        A = Tridiagonal(lower_diag, main_diag, upper_diag)
        C[i, j, :] .= A \ C_advected
    end
end

end # module VerticalTransportModule