# src/VerticalTransport.jl

module VerticalTransportModule

export vertical_transport!

using ..ModelStructs
using LinearAlgebra # For the efficient Tridiagonal solver

"""
    vertical_transport!(state::State, grid::Grid, dt::Float64)

Updates tracer concentrations due to vertical advection and diffusion.

This function iterates through each water column (i, j) and solves the
1D vertical transport equation. It uses an explicit first-order upwind
scheme for advection and a numerically stable implicit scheme for diffusion.
"""
function vertical_transport!(state::State, grid::Grid, dt::Float64)
    # For now, we assume a constant vertical diffusion coefficient.
    # In a more complex model, this could be a 3D field from the state.
    Kz = 1e-4 # m²/s, typical vertical eddy diffusivity

    # Loop over each tracer in the state
    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        solve_vertical_column!(C, state.w, grid, dt, Kz)
    end
    return nothing
end


"""
Helper function to solve the 1D advection-diffusion equation for all vertical columns.
"""
function solve_vertical_column!(C, w, grid, dt, Kz)
    nx, ny, nz = grid.dims
    
    # Loop over each horizontal (i, j) position
    for j in 1:ny, i in 1:nx
        
        # Explicit Advection Step
        # Calculate advective fluxes and create a temporary concentration field
        C_advected = C[i, j, :] # Extract the current column
        flux_z = zeros(nz + 1)

        for k in 2:nz
            velocity = w[i, j, k]
            if velocity >= 0 # Upward flow
                concentration_at_face = C[i, j, k-1]
            else # Downward flow
                concentration_at_face = C[i, j, k]
            end
            flux_z[k] = velocity * concentration_at_face * grid.face_area_z[i, j, k]
        end
        # We assume zero flux at the surface (k=nz+1) and bottom (k=1) for advection.
        
        for k in 1:nz
            flux_divergence = flux_z[k+1] - flux_z[k]
            if grid.volume[i, j, k] > 0
                C_advected[k] -= (dt / grid.volume[i, j, k]) * flux_divergence
            end
        end

        # Implicit Diffusion Step 
        # Now solve the diffusion equation: dC/dt = d/dz(Kz * dC/dz)
        # This results in a tridiagonal system `A * C_new = C_advected`

        # Get vertical grid spacing for this column
        dz = [grid.volume[i,j,k] / grid.face_area_z[i,j,k] for k in 1:nz]
        
        # Coefficients for the tridiagonal matrix
        alpha = Kz * dt ./ (dz .* dz)
        
        # Diagonals of the matrix `A`
        lower_diag = -alpha[2:end]
        main_diag  = 1.0 .+ 2.0 .* alpha
        upper_diag = -alpha[1:end-1]

        # No-flux boundary conditions by adjusting the main diagonal
        main_diag[1] = 1.0 + alpha[1]
        main_diag[end] = 1.0 + alpha[end]

        # Create the tridiagonal matrix
        A = Tridiagonal(lower_diag, main_diag, upper_diag)
        
        # Solve the system `A * C_new = d` where `d` is the result of the advection step
        C_new = A \ C_advected
        
        # Update the original concentration array with the new values for this column
        C[i, j, :] .= C_new
    end
end

end # module VerticalTransportModule