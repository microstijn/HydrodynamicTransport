# src/HorizontalTransportModule.jl

module HorizontalTransportModule

export horizontal_transport!

using ..ModelStructs
using StaticArrays

# --- Helper functions for 4th-Order Bott Scheme (from the stable version) ---

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

# --- Main Dispatcher Function (High-Performance Version) ---
function horizontal_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kh = 1.0 
    
    for tracer_name in keys(state.tracers)
        C_current = state.tracers[tracer_name]
        C_temp = similar(C_current)
        C_in = similar(C_current)

        advect_x!(C_temp, C_current, state.u, grid, dt)
        advect_y!(C_current, C_temp, state.v, grid, dt)
        
        # After advection, C_current has the result. It is the input for diffusion.
        copyto!(C_in, C_current)
        diffuse_x!(C_temp, C_in, grid, dt, Kh)
        copyto!(C_in, C_temp)
        diffuse_y!(C_current, C_in, grid, dt, Kh)
    end
    return nothing
end

# --- MONOTONIC, STABLE, MUTATING Advection Implementation ---

function advect_x!(C_out, C_in, u, grid::AbstractGrid, dt)
    nx, ny, nz = get_grid_dims(grid)
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    
    for k in 1:nz, j in 1:ny, i in 2:nx
        velocity = u[i, j, k]
        if abs(velocity) < 1e-12; continue; end

        donor_idx, receiver_idx = velocity >= 0 ? (i - 1, i) : (i, i - 1)
        
        c_stencil = get_stencil_x(C_in, donor_idx, j, k, nx)
        dx_donor = get_dx_at_face(grid, i, j)
        
        a0, a1, a2, a3, a4 = calculate_bott_coeffs(c_stencil...)
        courant_abs = abs(velocity * dt / dx_donor)
        if courant_abs > 1.0; @warn "Courant > 1"; courant_abs = 1.0; end

        if velocity >= 0
            integral_right = _indefinite_integral_poly4(0.5, a0, a1, a2, a3, a4)
            integral_left = _indefinite_integral_poly4(0.5 - courant_abs, a0, a1, a2, a3, a4)
        else
            integral_right = _indefinite_integral_poly4(-0.5 + courant_abs, a0, a1, a2, a3, a4)
            integral_left = _indefinite_integral_poly4(-0.5, a0, a1, a2, a3, a4)
        end
        integrated_area = integral_right - integral_left
        
        C_face_high_order = courant_abs > 1e-9 ? integrated_area / courant_abs : C_in[donor_idx, j, k]

        C_face_low_order = C_in[donor_idx, j, k]
        
        local r_numerator, r_denominator
        if velocity >= 0
            c_up_far = C_in[max(1, i-2), j, k]; c_up_near = C_in[i-1, j, k]; c_down_near = C_in[i, j, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        else
            c_up_far = C_in[min(nx, i+1), j, k]; c_up_near = C_in[i, j, k]; c_down_near = C_in[i-1, j, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        end
        
        local r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator
        
        phi = (r + abs(r)) / (1 + abs(r) + 1e-9)
        
        C_face_limited = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)

        C_max = max(C_in[donor_idx, j, k], C_in[receiver_idx, j, k])
        C_min = min(C_in[donor_idx, j, k], C_in[receiver_idx, j, k])
        C_face_limited = max(C_min, min(C_max, C_face_limited))
        
        fluxes_x[i, j, k] = velocity * C_face_limited * grid.face_area_x[i, j, k]
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        if grid.volume[i, j, k] > 1e-12
            flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
            C_out[i, j, k] = C_in[i,j,k] - (dt / grid.volume[i, j, k]) * flux_divergence
        else
            C_out[i, j, k] = C_in[i,j,k]
        end
    end
end

function advect_y!(C_out, C_in, v, grid::AbstractGrid, dt)
    nx, ny, nz = get_grid_dims(grid)
    fluxes_y = zeros(Float64, nx, ny + 1, nz)

    for k in 1:nz, i in 1:nx, j in 2:ny
        velocity = v[i, j, k]
        if abs(velocity) < 1e-12; continue; end

        donor_idx, receiver_idx = velocity >= 0 ? (j - 1, j) : (j, j - 1)
        
        c_stencil = get_stencil_y(C_in, i, donor_idx, k, ny)
        dy_donor = get_dy_at_face(grid, i, j)
        
        a0, a1, a2, a3, a4 = calculate_bott_coeffs(c_stencil...)
        courant_abs = abs(velocity * dt / dy_donor)
        if courant_abs > 1.0; @warn "Courant > 1"; courant_abs = 1.0; end

        if velocity >= 0
            integral_right = _indefinite_integral_poly4(0.5, a0, a1, a2, a3, a4)
            integral_left = _indefinite_integral_poly4(0.5 - courant_abs, a0, a1, a2, a3, a4)
        else
            integral_right = _indefinite_integral_poly4(-0.5 + courant_abs, a0, a1, a2, a3, a4)
            integral_left = _indefinite_integral_poly4(-0.5, a0, a1, a2, a3, a4)
        end
        integrated_area = integral_right - integral_left

        C_face_high_order = courant_abs > 1e-9 ? integrated_area / courant_abs : C_in[i, donor_idx, k]
        
        C_face_low_order = C_in[i, donor_idx, k]
        
        local r_numerator, r_denominator
        if velocity >= 0
            c_up_far = C_in[i, max(1, j-2), k]; c_up_near = C_in[i, j-1, k]; c_down_near = C_in[i, j, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        else
            c_up_far = C_in[i, min(ny, j+1), k]; c_up_near = C_in[i, j, k]; c_down_near = C_in[i, j-1, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        end

        local r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator

        phi = (r + abs(r)) / (1 + abs(r) + 1e-9)

        C_face_limited = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)

        C_max = max(C_in[i, donor_idx, k], C_in[i, receiver_idx, k])
        C_min = min(C_in[i, donor_idx, k], C_in[i, receiver_idx, k])
        C_face_limited = max(C_min, min(C_max, C_face_limited))
        
        fluxes_y[i, j, k] = velocity * C_face_limited * grid.face_area_y[i, j, k]
    end

    for k in 1:nz, i in 1:nx, j in 1:ny
        if grid.volume[i, j, k] > 1e-12
            flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
            C_out[i, j, k] = C_in[i,j,k] - (dt / grid.volume[i, j, k]) * flux_divergence
        else
            C_out[i, j, k] = C_in[i,j,k]
        end
    end
end

# --- STABLE, MUTATING Diffusion Implementation ---
function diffuse_x!(C_out, C_in, grid, dt, Kh)
    nx, ny, nz = get_grid_dims(grid)
    fluxes_x = zeros(Float64, nx + 1, ny, nz)
    
    for k in 1:nz, j in 1:ny, i in 2:nx
        dx = get_dx_centers(grid, i, j)
        dCdx = (C_in[i, j, k] - C_in[i-1, j, k]) / dx
        fluxes_x[i, j, k] = -Kh * grid.face_area_x[i, j, k] * dCdx
    end
    for k in 1:nz, j in 1:ny, i in 1:nx
        if grid.volume[i, j, k] > 1e-12
            flux_divergence = fluxes_x[i+1, j, k] - fluxes_x[i, j, k]
            # --- FINAL CORRECT IMPLEMENTATION ---
            C_out[i, j, k] = C_in[i, j, k] - (dt / grid.volume[i, j, k]) * flux_divergence
        else
            C_out[i,j,k] = C_in[i,j,k]
        end
    end
end

function diffuse_y!(C_out, C_in, grid, dt, Kh)
    nx, ny, nz = get_grid_dims(grid)
    fluxes_y = zeros(Float64, nx, ny + 1, nz)
    
    for k in 1:nz, i in 1:nx, j in 2:ny
        dy = get_dy_centers(grid, i, j)
        dCdy = (C_in[i, j, k] - C_in[i, j-1, k]) / dy
        fluxes_y[i, j, k] = -Kh * grid.face_area_y[i, j, k] * dCdy
    end
    for k in 1:nz, i in 1:nx, j in 1:ny
        if grid.volume[i, j, k] > 1e-12
            flux_divergence = fluxes_y[i, j+1, k] - fluxes_y[i, j, k]
            # --- FINAL CORRECT IMPLEMENTATION ---
            C_out[i, j, k] = C_in[i, j, k] - (dt / grid.volume[i, j, k]) * flux_divergence
        else
            C_out[i,j,k] = C_in[i,j,k]
        end
    end
end

# --- Helpers for Grid Dimensions and Spacing ---
get_grid_dims(grid::CartesianGrid) = Tuple(grid.dims)
get_grid_dims(grid::CurvilinearGrid) = (grid.nx, grid.ny, grid.nz)

get_dx_at_face(grid::CartesianGrid, i, j) = grid.x[2,1,1] - grid.x[1,1,1]
get_dx_at_face(grid::CurvilinearGrid, i, j) = 1.0 / grid.pm[min(i, grid.nx), j]

get_dy_at_face(grid::CartesianGrid, i, j) = grid.y[1,2,1] - grid.y[1,1,1]
get_dy_at_face(grid::CurvilinearGrid, i, j) = 1.0 / grid.pn[i, min(j, grid.ny)]

get_dx_centers(grid::CartesianGrid, i, j) = grid.x[2,1,1] - grid.x[1,1,1]
get_dx_centers(grid::CurvilinearGrid, i, j) = 1 / (0.5 * (grid.pm[i-1, j] + grid.pm[i, j]))

get_dy_centers(grid::CartesianGrid, i, j) = grid.y[1,2,1] - grid.y[1,1,1]
get_dy_centers(grid::CurvilinearGrid, i, j) = 1 / (0.5 * (grid.pn[i, j-1] + grid.pn[i, j]))

end # module HorizontalTransportModule