# src/VerticalTransport.jl

module VerticalTransportModule

export vertical_transport!

using ..ModelStructs
using LinearAlgebra # For the efficient Tridiagonal solver

"""
    vertical_transport!(state::State, grid::Grid, dt::Float64)

Updates tracer concentrations due to vertical advection and diffusion on the staggered grid.
"""
function vertical_transport!(state::State, grid::Grid, dt::Float64)
    Kz = 1e-4 # m²/s, typical vertical eddy diffusivity

    # Loop over each tracer in the state
    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        solve_vertical_column!(C, state.w, grid, dt, Kz)
    end
    return nothing
end


function solve_vertical_column!(C, w, grid, dt, Kz)
    nx, ny, nz = grid.dims
    if nz <= 1
        return nothing
    end

    for j in 1:ny, i in 1:nx
        # Extract the current column
        current_C_column = C[i, j, :]
        C_advected = copy(current_C_column)
        flux_z = zeros(nz + 1)

        # Explicit Advection Step
        # Loop over interior z-faces (k=2 to nz)
        for k in 2:nz
            velocity = w[i, j, k] # This `w` is now correctly located on the face `k`
            if velocity >= 0 # Upward flow
                concentration_at_face = C[i, j, k-1]
            else # Downward flow
                concentration_at_face = C[i, j, k]
            end
            # face_area_z is correctly sized for staggered grid
            flux_z[k] = velocity * concentration_at_face * grid.face_area_z[i, j, k]
        end
        # Assume zero flux at the surface (k=nz+1) and bottom (k=1)
        for k in 1:nz
            flux_divergence = flux_z[k+1] - flux_z[k]
            if grid.volume[i, j, k] > 0
                C_advected[k] -= (dt / grid.volume[i, j, k]) * flux_divergence
            end
        end

        # Implicit Diffusion Step
        if nz > 1
            dz_column = [grid.volume[i,j,k] / grid.face_area_z[i,j,k] for k in 1:nz]
            alpha = (Kz * dt) ./ (dz_column .^ 2)
            
            lower_diag = -alpha[2:end]
            main_diag  = 1.0 .+ 2.0 .* alpha
            upper_diag = -alpha[1:end-1]

            main_diag[1] = 1.0 + alpha[1]
            main_diag[end] = 1.0 + alpha[end]

            A = Tridiagonal(lower_diag, main_diag, upper_diag)
            C_new_column = A \ C_advected
            
            C[i, j, :] .= C_new_column
        end
    end
end

end # module VerticalTransportModule