# src/HorizontalTransportModule.jl

module HorizontalTransportModule

export horizontal_transport!

using ..ModelStructs

# --- Helper functions (defined at the top) ---
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

# --- Main Public Function (Dispatcher) ---
function horizontal_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kh = 5.0 
    for tracer_name in keys(state.tracers)
        C = state.tracers[tracer_name]
        advect_x!(C, state.u, grid, dt)
        advect_y!(C, state.v, grid, dt)
        diffuse_x!(C, grid, dt, Kh)
        diffuse_y!(C, grid, dt, Kh)
    end
    return nothing
end

# --- Advection Methods for CartesianGrid (Unchanged) ---
function advect_x!(C::Array{Float64, 3}, u::Array{Float64, 3}, grid::CartesianGrid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)

    for k in 1:nz, j in 1:ny
        vel_left = u[1, j, k]; vel_right = u[nx+1, j, k]
        if vel_left < 0; fluxes_x[1, j, k] = vel_left * C[1, j, k] * grid.face_area_x[1, j, k] * dt; end
        if vel_right > 0; fluxes_x[nx+1, j, k] = vel_right * C[nx, j, k] * grid.face_area_x[nx+1, j, k] * dt; end
    end

    for k in 1:nz, j in 1:ny, i in 2:nx
        velocity = u[i, j, k]
        if i <= 2 || i >= nx
            fluxes_x[i, j, k] = calculate_fou_flux(velocity, C[i-1, j, k], C[i, j, k], grid.face_area_x[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? i - 1 : i
            c_stencil = get_stencil_x(C, upwind_idx, j, k, nx)
            dx_donor = grid.volume[upwind_idx, j, k] / grid.face_area_y[upwind_idx, j, k]
            C_donor = C[upwind_idx, j, k]; volume_donor = grid.volume[upwind_idx, j, k]
            fluxes_x[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dx_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
        if grid.volume[i, j, k] > 0; C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]; end
    end
    C .= C_new
end

function advect_y!(C::Array{Float64, 3}, v::Array{Float64, 3}, grid::CartesianGrid, dt::Float64)
    nx, ny, nz = grid.dims
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)

    for k in 1:nz, i in 1:nx
        vel_bottom = v[i, 1, k]; vel_top = v[i, ny+1, k]
        if vel_bottom < 0; fluxes_y[i, 1, k] = vel_bottom * C[i, 1, k] * grid.face_area_y[i, 1, k] * dt; end
        if vel_top > 0; fluxes_y[i, ny+1, k] = vel_top * C[i, ny, k] * grid.face_area_y[i, ny+1, k] * dt; end
    end

    for k in 1:nz, i in 1:nx, j in 2:ny
        velocity = v[i, j, k]
        if j <= 2 || j >= ny
            fluxes_y[i, j, k] = calculate_fou_flux(velocity, C[i, j-1, k], C[i, j, k], grid.face_area_y[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? j - 1 : j
            c_stencil = get_stencil_y(C, i, upwind_idx, k, ny)
            dy_donor = grid.volume[i, upwind_idx, k] / grid.face_area_x[i, upwind_idx, k]
            C_donor = C[i, upwind_idx, k]; volume_donor = grid.volume[i, upwind_idx, k]
            fluxes_y[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dy_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, i in 1:nx, j in 1:ny
        flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
        if grid.volume[i, j, k] > 0; C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]; end
    end
    C .= C_new
end


# --- Advection Methods for CurvilinearGrid ---

function advect_x!(C::Array{Float64, 3}, u::Array{Float64, 3}, grid::CurvilinearGrid, dt::Float64)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)

    # --- FIX: Add open boundary conditions for the curvilinear grid ---
    for k in 1:nz, j in 1:ny
        # Left boundary (i=1)
        vel_left = u[1, j, k]
        if vel_left < 0 # Outward flow
            fluxes_x[1, j, k] = vel_left * C[1, j, k] * grid.face_area_x[1, j, k] * dt
        end
        # Right boundary (i=nx+1)
        vel_right = u[nx+1, j, k]
        if vel_right > 0 # Outward flow
            fluxes_x[nx+1, j, k] = vel_right * C[nx, j, k] * grid.face_area_x[nx+1, j, k] * dt
        end
    end

    # Interior fluxes (using Bott scheme)
    for k in 1:nz, j in 1:ny, i in 2:nx
        velocity = u[i, j, k]
        if i <= 2 || i >= nx
            fluxes_x[i, j, k] = calculate_fou_flux(velocity, C[i-1, j, k], C[i, j, k], grid.face_area_x[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? i - 1 : i
            c_stencil = get_stencil_x(C, upwind_idx, j, k, nx)
            dx_donor = 1 / grid.pm[upwind_idx, j]
            C_donor = C[upwind_idx, j, k]
            volume_donor = grid.volume[upwind_idx, j, k]
            fluxes_x[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dx_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
        if grid.volume[i, j, k] > 0; C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]; end
    end
    C .= C_new
end

function advect_y!(C::Array{Float64, 3}, v::Array{Float64, 3}, grid::CurvilinearGrid, dt::Float64)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)

    # --- FIX: Add open boundary conditions for the curvilinear grid ---
    for k in 1:nz, i in 1:nx
        # Bottom boundary (j=1)
        vel_bottom = v[i, 1, k]
        if vel_bottom < 0 # Outward flow
            fluxes_y[i, 1, k] = vel_bottom * C[i, 1, k] * grid.face_area_y[i, 1, k] * dt
        end
        # Top boundary (j=ny+1)
        vel_top = v[i, ny+1, k]
        if vel_top > 0 # Outward flow
            fluxes_y[i, ny+1, k] = vel_top * C[i, ny, k] * grid.face_area_y[i, ny+1, k] * dt
        end
    end

    # Interior fluxes (using Bott scheme)
    for k in 1:nz, i in 1:nx, j in 2:ny
        velocity = v[i, j, k]
        if j <= 2 || j >= ny
            fluxes_y[i, j, k] = calculate_fou_flux(velocity, C[i, j-1, k], C[i, j, k], grid.face_area_y[i,j,k], dt)
        else
            upwind_idx = velocity >= 0 ? j - 1 : j
            c_stencil = get_stencil_y(C, i, upwind_idx, k, ny)
            dy_donor = 1 / grid.pn[i, upwind_idx]
            C_donor = C[i, upwind_idx, k]
            volume_donor = grid.volume[i, upwind_idx, k]
            fluxes_y[i, j, k] = calculate_bott_flux(c_stencil, velocity, dt, dy_donor, C_donor, volume_donor)
        end
    end

    for k in 1:nz, i in 1:nx, j in 1:ny
        flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
        if grid.volume[i, j, k] > 0; C_new[i, j, k] -= flux_divergence / grid.volume[i, j, k]; end
    end
    C .= C_new
end


# --- Diffusion Methods (Unchanged) ---
function diffuse_x!(C::Array{Float64, 3}, grid::AbstractGrid, dt::Float64, Kh::Float64)
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    C_new = copy(C)
    
    for k in 1:nz, j in 1:ny, i in 2:nx
        # Use multiple dispatch to get the correct grid spacing
        dx_centers = get_dx_centers(grid, i, j, k)
        dCdx = (C[i, j, k] - C[i-1, j, k]) / dx_centers
        fluxes_x[i, j, k] = -Kh * grid.face_area_x[i, j, k] * dCdx
    end

    volume = isa(grid, CartesianGrid) ? grid.volume : grid.volume
    for k in 1:nz, j in 1:ny, i in 1:nx
        flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
        if volume[i, j, k] > 0; C_new[i, j, k] -= (dt / volume[i, j, k]) * flux_divergence; end
    end
    C .= C_new
end

get_dx_centers(grid::CartesianGrid, i, j, k) = grid.x[2,1,1] - grid.x[1,1,1]
get_dx_centers(grid::CurvilinearGrid, i, j, k) = 1 / (0.5 * (grid.pm[i-1, j] + grid.pm[i, j]))

function diffuse_y!(C::Array{Float64, 3}, grid::AbstractGrid, dt::Float64, Kh::Float64)
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    C_new = copy(C)
    
    for k in 1:nz, i in 1:nx, j in 2:ny
        dy_centers = get_dy_centers(grid, i, j, k)
        dCdy = (C[i, j, k] - C[i, j-1, k]) / dy_centers
        fluxes_y[i, j, k] = -Kh * grid.face_area_y[i, j, k] * dCdy
    end

    volume = isa(grid, CartesianGrid) ? grid.volume : grid.volume
    for k in 1:nz, i in 1:nx, j in 1:ny
        flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
        if volume[i, j, k] > 0; C_new[i, j, k] -= (dt / volume[i, j, k]) * flux_divergence; end
    end
    C .= C_new
end

get_dy_centers(grid::CartesianGrid, i, j, k) = grid.y[1,2,1] - grid.y[1,1,1]
get_dy_centers(grid::CurvilinearGrid, i, j, k) = 1 / (0.5 * (grid.pn[i, j-1] + grid.pn[i, j]))

end # module HorizontalTransportModule