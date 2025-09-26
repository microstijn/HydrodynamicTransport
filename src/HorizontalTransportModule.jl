
module HorizontalTransportModule

export horizontal_transport!

using ..ModelStructs

"""
    horizontal_transport!(state::State, grid::Grid, dt::Float64)

Updates tracer concentrations due to horizontal advection.
This function uses the 4th-Order Bott (1989) scheme for the interior of the domain
and falls back to the 1st-Order Upwind (FOU) scheme at the boundaries.
"""
function horizontal_transport!(state::State, grid::Grid, dt::Float64)
    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        
        advect_x!(C, state.u, grid, dt)
        advect_y!(C, state.v, grid, dt)
    end
    return nothing
end

# Coefficient and Stencil Helpers

function get_stencil_x(C::Array{Float64, 3}, i::Int, j::Int, k::Int, nx::Int)
    im2 = max(1, i - 2); im1 = max(1, i - 1)
    ip1 = min(nx, i + 1); ip2 = min(nx, i + 2)
    return C[im2, j, k], C[im1, j, k], C[i, j, k], C[ip1, j, k], C[ip2, j, k]
end

function get_stencil_y(C::Array{Float64, 3}, i::Int, j::Int, k::Int, ny::Int)
    jm2 = max(1, j - 2); jm1 = max(1, j - 1)
    jp1 = min(ny, j + 1); jp2 = min(ny, j + 2)
    return C[i, jm2, k], C[i, jm1, k], C[i, j, k], C[i, jp1, k], C[i, jp2, k]
end


"""
    calculate_bott_coeffs(c_im2, c_im1, c_i, c_ip1, c_ip2)

Calculates the 5 coefficients of the 4th-order polynomial for a cell.
"""
function calculate_bott_coeffs(c_im2, c_im1, c_i, c_ip1, c_ip2)
    a1 = (c_ip1 - c_im1) / 2.0
    a3 = (c_ip2 - 2*c_ip1 + 2*c_im1 - c_im2) / 12.0 - (2/3.0) * a1
    a4 = (c_ip2 - 4*c_ip1 + 6*c_i - 4*c_im1 + c_im2) / 24.0
    a0 = c_i - a4
    a2 = a4

    return a0, a1, a2, a3, a4
end


#Flux Calculation Helpers

"""
    calculate_fou_flux(velocity, C_up, C_down, face_area)

Calculates the flux across a cell face using the First-Order Upwind scheme.
"""
function calculate_fou_flux(velocity::Float64, C_up::Float64, C_down::Float64, face_area::Float64)
    if velocity >= 0
        # Flow is positive, use concentration from the "up" cell (e.g., left or bottom)
        return velocity * C_up * face_area
    else
        # Flow is negative, use concentration from the "down" cell (e.g., right or top)
        return velocity * C_down * face_area
    end
end

"""
    calculate_bott_flux(c_stencil, velocity, dt, cell_volume, face_area, C_donor)

Calculates the final, limited flux across a cell face using the Bott scheme.
"""
function calculate_bott_flux(c_stencil::NTuple{5, Float64}, velocity::Float64, dt::Float64, dx::Float64, C_donor::Float64, volume_donor::Float64)
    # Step 1: Get polynomial coefficients
    a0, a1, a2, a3, a4 = calculate_bott_coeffs(c_stencil...)

    # Step 2: Calculate high-order concentration (I_plus)
    courant = velocity * dt / dx
    c_abs = abs(courant)
    I_plus_concentration = c_abs * (a0 + c_abs/2 * (a1 + c_abs/3 * (a2 + c_abs/4 * (a3 + c_abs/5 * a4))))
    
    # Convert the high-order concentration to a high-order MASS
    M_plus_mass = I_plus_concentration * volume_donor

    # Step 3: Apply Positivity Limiter (comparing MASS to MASS)
    mass_in_donor_cell = C_donor * volume_donor
    M_limited = max(0.0, min(M_plus_mass, mass_in_donor_cell))
    
    return sign(velocity) * M_limited
end



"""
    advect_x!(C, u, grid, dt)

Calculates advection in the x-direction using a mixed Bott/FOU scheme.
"""
function advect_x!(C::Array{Float64, 3}, u::Array{Float64, 3}, grid::Grid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)

    for k in 1:nz, j in 1:ny
        for i in 2:nx
            velocity = u[i, j, k]
            if i < 3 || i > nx - 1
                fluxes_x[i, j, k] = calculate_fou_flux(velocity, C[i-1, j, k], C[i, j, k], grid.face_area_x[i, j, k])
            else
                upwind_idx = velocity >= 0 ? i - 1 : i
                c_stencil = get_stencil_x(C, upwind_idx, j, k, nx)
                dx = grid.volume[upwind_idx, j, k] / grid.face_area_x[upwind_idx, j, k]
                C_donor = C[upwind_idx, j, k]
                volume_donor = grid.volume[upwind_idx, j, k]
                fluxes_x[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dx, C_donor, volume_donor)
            end
        end
        for i in 1:nx
            flux_div = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
            if grid.volume[i, j, k] > 0
                C_new[i, j, k] -= flux_div / grid.volume[i, j, k]
            end
        end
    end
    C .= C_new
end

function advect_y!(C::Array{Float64, 3}, v::Array{Float64, 3}, grid::Grid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)

    for k in 1:nz, i in 1:nx
        for j in 2:ny
            velocity = v[i, j, k]
            if j < 3 || j > ny - 1
                fluxes_y[i, j, k] = calculate_fou_flux(velocity, C[i, j-1, k], C[i, j, k], grid.face_area_y[i, j, k])
            else
                upwind_idx = velocity >= 0 ? j - 1 : j
                c_stencil = get_stencil_y(C, i, upwind_idx, k, ny)
                dy = grid.volume[i, upwind_idx, k] / grid.face_area_y[i, upwind_idx, k]
                C_donor = C[i, upwind_idx, k]
                volume_donor = grid.volume[i, upwind_idx, k]
                fluxes_y[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dy, C_donor, volume_donor)
            end
        end
        for j in 1:ny
            flux_div = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
            if grid.volume[i, j, k] > 0
                C_new[i, j, k] -= flux_div / grid.volume[i, j, k]
            end
        end
    end
    C .= C_new
end

end # module HorizontalTransportModule