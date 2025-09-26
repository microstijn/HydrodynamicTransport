module HorizontalTransportModule

export horizontal_transport!

using ..ModelStructs


"""
    horizontal_transport!(state::State, grid::Grid, dt::Float64)

Updates all tracer concentrations due to horizontal advection and diffusion.
"""
function horizontal_transport!(state::State, grid::Grid, dt::Float64)
    # Increased diffusivity to make the effect more visible in visualization
    Kh = 5.0 # TODO change to be fixed value dependend on gridsize

    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        
        # Operator Splitting: Advection first, then diffusion
        advect_x!(C, state.u, grid, dt)
        advect_y!(C, state.v, grid, dt)
        
        diffuse_x!(C, grid, dt, Kh)
        diffuse_y!(C, grid, dt, Kh)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Advection Code (Unchanged)
# -----------------------------------------------------------------------------

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

function calculate_bott_coeffs(c_im2, c_im1, c_i, c_ip1, c_ip2)
    a1 = (c_ip1 - c_im1) / 2.0
    a3 = (c_ip2 - 2*c_ip1 + 2*c_im1 - c_im2) / 12.0 - (2/3.0) * a1
    a4 = (c_ip2 - 4*c_ip1 + 6*c_i - 4*c_im1 + c_im2) / 24.0
    a0 = c_i - a4; a2 = a4
    return a0, a1, a2, a3, a4
end

function _indefinite_integral_poly4(xi, a0, a1, a2, a3, a4)
    return xi * (a0 + xi/2 * (a1 + xi/3 * (a2 + xi/4 * (a3 + xi/5 * a4))))
end

function calculate_fou_flux(velocity::Float64, C_up::Float64, C_down::Float64, face_area::Float64, dt::Float64)
    mass = velocity >= 0 ? velocity * C_up * face_area * dt : velocity * C_down * face_area * dt
    return mass
end

function calculate_bott_flux(c_stencil::NTuple{5, Float64}, velocity::Float64, dt::Float64, dx::Float64, C_donor::Float64, volume_donor::Float64)
    a0, a1, a2, a3, a4 = calculate_bott_coeffs(c_stencil...)
    courant = velocity * dt / dx
    c_abs = abs(courant)
    
    local integrated_mass_norm
    if velocity >= 0
        integral_right = _indefinite_integral_poly4(0.5, a0, a1, a2, a3, a4)
        integral_left = _indefinite_integral_poly4(0.5 - c_abs, a0, a1, a2, a3, a4)
        integrated_mass_norm = integral_right - integral_left
    else
        integral_right = _indefinite_integral_poly4(-0.5 + c_abs, a0, a1, a2, a3, a4)
        integral_left = _indefinite_integral_poly4(-0.5, a0, a1, a2, a3, a4)
        integrated_mass_norm = integral_right - integral_left
    end

    M_high_order = integrated_mass_norm * volume_donor
    mass_in_donor_cell = C_donor * volume_donor
    M_limited_magnitude = max(0.0, min(abs(M_high_order), mass_in_donor_cell))
    
    return sign(velocity) * M_limited_magnitude
end

function advect_x!(C::Array{Float64, 3}, u::Array{Float64, 3}, grid::Grid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)

    for k in 1:nz, j in 1:ny, i in 2:nx
        velocity = u[i, j, k]
        if i <= 2 || i >= nx
            fluxes_x[i, j, k] = calculate_fou_flux(velocity, C[i-1, j, k], C[i, j, k], grid.face_area_x[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? i - 1 : i
            c_stencil = get_stencil_x(C, upwind_idx, j, k, nx)
            dx_donor = grid.volume[upwind_idx, j, k] / grid.face_area_y[upwind_idx, j, k]
            C_donor = C[upwind_idx, j, k]
            volume_donor = grid.volume[upwind_idx, j, k]
            fluxes_x[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dx_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
        if grid.volume[i, j, k] > 0
            C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]
        end
    end
    C .= C_new
end

function advect_y!(C::Array{Float64, 3}, v::Array{Float64, 3}, grid::Grid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)

    for k in 1:nz, i in 1:nx, j in 2:ny
        velocity = v[i, j, k]
        if j <= 2 || j >= ny
            fluxes_y[i, j, k] = calculate_fou_flux(velocity, C[i, j-1, k], C[i, j, k], grid.face_area_y[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? j - 1 : j
            c_stencil = get_stencil_y(C, i, upwind_idx, k, ny)
            dy_donor = grid.volume[i, upwind_idx, k] / grid.face_area_x[i, upwind_idx, k]
            C_donor = C[i, upwind_idx, k]
            volume_donor = grid.volume[i, upwind_idx, k]
            fluxes_y[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dy_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, i in 1:nx, j in 1:ny
        flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
        if grid.volume[i, j, k] > 0
            C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]
        end
    end
    C .= C_new
end


# -----------------------------------------------------------------------------
# Horizontal Diffusion Code
# -----------------------------------------------------------------------------

"""
    diffuse_x!(C, grid, dt, Kh)

Solves the diffusion equation in the x-direction.
"""
function diffuse_x!(C::Array{Float64, 3}, grid::Grid, dt::Float64, Kh::Float64)
    nx, ny, nz = grid.dims
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)
    
    for k in 1:nz, j in 1:ny, i in 2:nx
        dx = (grid.x[i, j, k] - grid.x[i-1, j, k])
        dCdx = (C[i, j, k] - C[i-1, j, k]) / dx
        fluxes_x[i, j, k] = -Kh * grid.face_area_x[i, j, k] * dCdx
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
        if grid.volume[i, j, k] > 0
            # --- BUG FIX ---
            # The change in concentration is SUBTRACTED, not added.
            # dC/dt = -∇·F. The divergence is (F_out - F_in).
            C_new[i, j, k] -= (dt / grid.volume[i, j, k]) * flux_divergence
        end
    end
    C .= C_new
end

"""
    diffuse_y!(C, grid, dt, Kh)

Solves the diffusion equation in the y-direction.
"""
function diffuse_y!(C::Array{Float64, 3}, grid::Grid, dt::Float64, Kh::Float64)
    nx, ny, nz = grid.dims
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)

    for k in 1:nz, i in 1:nx, j in 2:ny
        dy = (grid.y[i, j, k] - grid.y[i, j-1, k])
        dCdy = (C[i, j, k] - C[i, j-1, k]) / dy
        fluxes_y[i, j, k] = -Kh * grid.face_area_y[i, j, k] * dCdy
    end

    for k in 1:nz, i in 1:nx, j in 1:ny
        flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
        if grid.volume[i, j, k] > 0
            # --- BUG FIX ---
            # The change in concentration is SUBTRACTED, not added.
            C_new[i, j, k] -= (dt / grid.volume[i, j, k]) * flux_divergence
        end
    end
    C .= C_new
end

end # module HorizontalTransportModule

