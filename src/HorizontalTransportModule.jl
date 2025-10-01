# src/HorizontalTransportModule.jl

module HorizontalTransportModule

export horizontal_transport!

using ..HydrodynamicTransport.ModelStructs
using StaticArrays

# --- Stencil Functions ---
function get_stencil_x(C::Array{Float64, 3}, i_glob::Int, j_glob::Int, k::Int)
    return C[i_glob-2, j_glob, k], C[i_glob-1, j_glob, k], C[i_glob, j_glob, k], C[i_glob+1, j_glob, k], C[i_glob+2, j_glob, k]
end

function get_stencil_y(C::Array{Float64, 3}, i_glob::Int, j_glob::Int, k::Int)
    return C[i_glob, j_glob-2, k], C[i_glob, j_glob-1, k], C[i_glob, j_glob, k], C[i_glob, j_glob+1, k], C[i_glob, j_glob+2, k]
end

# --- Helper functions for Bott Scheme ---
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

# --- Main Dispatcher Function ---
function horizontal_transport!(state::State, grid::AbstractGrid, dt::Float64)
    Kh = 1.0 
    for tracer_name in keys(state.tracers)
        C_current = state.tracers[tracer_name]
        # C_in has already been prepared with ghost cells filled by BoundaryConditionsModule
        C_in = C_current
        
        C_temp = deepcopy(C_current)
        advect_x!(C_temp, C_in, state.u, grid, dt)
        advect_y!(C_current, C_temp, state.v, grid, dt)
        
        copyto!(C_in, C_current)
        diffuse_x!(C_temp, C_in, grid, dt, Kh)
        copyto!(C_in, C_temp)
        diffuse_y!(C_current, C_in, grid, dt, Kh)
    end
    return nothing
end

# --- STABLE HYBRID ADVECTION SCHEME ---
function advect_x!(C_out, C_in, u, grid::AbstractGrid, dt)
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    fluxes_x = zeros(size(u))

    # --- Deep Interior (High-Order TVD) ---
    # The 5-point stencil is safe to call when the donor cell is at least 3 cells away from the global edge.
    for k in axes(C_in, 3), j_phys in 1:ny, i_phys in 3:nx-1
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        velocity = u[i_glob, j_glob, k]
        if abs(velocity) < 1e-12; continue; end
        
        donor_idx = velocity >= 0 ? i_glob - 1 : i_glob
        receiver_idx = velocity >= 0 ? i_glob : i_glob - 1
        
        c_stencil = get_stencil_x(C_in, donor_idx, j_glob, k)
        dx_donor = get_dx_at_face(grid, i_phys, j_phys)
        
        a0,a1,a2,a3,a4 = calculate_bott_coeffs(c_stencil...)
        courant_abs = abs(velocity * dt / dx_donor)
        if courant_abs > 1.0; @warn "Courant > 1"; courant_abs = 1.0; end

        if velocity >= 0
            integral_right=_indefinite_integral_poly4(0.5, a0,a1,a2,a3,a4); integral_left=_indefinite_integral_poly4(0.5-courant_abs, a0,a1,a2,a3,a4)
        else
            integral_right=_indefinite_integral_poly4(-0.5+courant_abs, a0,a1,a2,a3,a4); integral_left=_indefinite_integral_poly4(-0.5, a0,a1,a2,a3,a4)
        end
        C_face_high_order = (integral_right - integral_left) / (courant_abs + 1e-12)
        C_face_low_order = C_in[donor_idx, j_glob, k]
        c_up_far=C_in[donor_idx - (velocity >= 0 ? 1 : -1), j_glob, k]; c_up_near=C_in[donor_idx, j_glob, k]; c_down_near=C_in[receiver_idx, j_glob, k]
        r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator
        phi = (r + abs(r)) / (1 + abs(r) + 1e-9)
        C_face_tvd = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)
        C_max = max(C_in[donor_idx, j_glob, k], C_in[receiver_idx, j_glob, k])
        C_min = min(C_in[donor_idx, j_glob, k], C_in[receiver_idx, j_glob, k])
        
        fluxes_x[i_glob, j_glob, k] = velocity * max(C_min, min(C_max, C_face_tvd)) * grid.face_area_x[i_glob, j_glob, k]
    end

    # --- Boundary-Adjacent Faces (1st-Order Upwind) ---
    for k in axes(C_in, 3), j_phys in 1:ny
        j_glob = j_phys + ng
        # West faces (i_phys=1,2)
        for i_phys in 1:2
            i_glob = i_phys + ng
            vel = u[i_glob, j_glob, k]
            fluxes_x[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob-1, j_glob, k] : C_in[i_glob, j_glob, k]) * grid.face_area_x[i_glob, j_glob, k]
        end

        # East faces (i_phys=nx, nx+1)
        for i_phys in nx:nx+1
            i_glob = i_phys + ng
            vel = u[i_glob, j_glob, k]
            fluxes_x[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob-1, j_glob, k] : C_in[i_glob, j_glob, k]) * grid.face_area_x[i_glob, j_glob, k]
        end
    end

    # Final update over physical domain
    for k in axes(C_out, 3), j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_x[i_glob+1, j_glob, k] - fluxes_x[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
    end
end

function advect_y!(C_out, C_in, v, grid::AbstractGrid, dt)
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    fluxes_y = zeros(size(v))

    # --- Deep Interior (High-Order TVD) ---
    for k in axes(C_in, 3), j_phys in 3:ny-1, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        velocity = v[i_glob, j_glob, k]
        if abs(velocity) < 1e-12; continue; end
        
        donor_idx = velocity >= 0 ? j_glob - 1 : j_glob
        receiver_idx = velocity >= 0 ? j_glob : j_glob - 1

        c_stencil = get_stencil_y(C_in, i_glob, donor_idx, k)
        dy_donor = get_dy_at_face(grid, i_phys, j_phys)
        
        a0,a1,a2,a3,a4 = calculate_bott_coeffs(c_stencil...)
        courant_abs = abs(velocity * dt / dy_donor)
        if courant_abs > 1.0; @warn "Courant > 1"; courant_abs = 1.0; end

        if velocity >= 0
            integral_right=_indefinite_integral_poly4(0.5, a0,a1,a2,a3,a4); integral_left=_indefinite_integral_poly4(0.5-courant_abs, a0,a1,a2,a3,a4)
        else
            integral_right=_indefinite_integral_poly4(-0.5+courant_abs, a0,a1,a2,a3,a4); integral_left=_indefinite_integral_poly4(-0.5, a0,a1,a2,a3,a4)
        end
        C_face_high_order = (integral_right - integral_left) / (courant_abs + 1e-12)
        C_face_low_order = C_in[i_glob, donor_idx, k]
        c_up_far=C_in[i_glob, donor_idx - (velocity >= 0 ? 1 : -1), k]; c_up_near=C_in[i_glob, donor_idx, k]; c_down_near=C_in[i_glob, receiver_idx, k]
        r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
        r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator
        phi = (r + abs(r)) / (1 + abs(r) + 1e-9)
        C_face_tvd = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)
        C_max = max(C_in[i_glob, donor_idx, k], C_in[i_glob, receiver_idx, k])
        C_min = min(C_in[i_glob, donor_idx, k], C_in[i_glob, receiver_idx, k])
        
        fluxes_y[i_glob, j_glob, k] = velocity * max(C_min, min(C_max, C_face_tvd)) * grid.face_area_y[i_glob, j_glob, k]
    end

    # --- Boundary-Adjacent Faces (1st-Order Upwind) ---
    for k in axes(C_in, 3), i_phys in 1:nx
        i_glob = i_phys + ng
        # South faces (j_phys=1,2)
        for j_phys in 1:2
            j_glob = j_phys + ng
            vel = v[i_glob, j_glob, k]
            fluxes_y[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob, j_glob-1, k] : C_in[i_glob, j_glob, k]) * grid.face_area_y[i_glob, j_glob, k]
        end
        # North faces (j_phys=ny, ny+1)
        for j_phys in ny:ny+1
            j_glob = j_phys + ng
            vel = v[i_glob, j_glob, k]
            fluxes_y[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob, j_glob-1, k] : C_in[i_glob, j_glob, k]) * grid.face_area_y[i_glob, j_glob, k]
        end
    end

    # Final update over physical domain
    for k in axes(C_out, 3), j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_y[i_glob, j_glob+1, k] - fluxes_y[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
    end
end

function diffuse_x!(C_out, C_in, grid, dt, Kh)
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    fluxes_x = zeros(size(C_in, 1) + 1, size(C_in, 2), size(C_in, 3))
    
    # Loop over all physical interior faces
    for k in axes(C_in, 3), j_phys in 1:ny, i_phys in 2:nx+1
        i_glob, j_glob = i_phys + ng, j_phys + ng
        dx = get_dx_centers(grid, i_phys, j_phys)
        dCdx = (C_in[i_glob, j_glob, k] - C_in[i_glob-1, j_glob, k]) / dx
        fluxes_x[i_glob, j_glob, k] = -Kh * grid.face_area_x[i_glob, j_glob, k] * dCdx
    end

    for k in axes(C_out, 3), j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_x[i_glob+1, j_glob, k] - fluxes_x[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
    end
end

function diffuse_y!(C_out, C_in, grid, dt, Kh)
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    ng = grid.ng
    fluxes_y = zeros(size(C_in, 1), size(C_in, 2) + 1, size(C_in, 3))
    
    for k in axes(C_in, 3), j_phys in 2:ny+1, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        dy = get_dy_centers(grid, i_phys, j_phys)
        dCdy = (C_in[i_glob, j_glob, k] - C_in[i_glob, j_glob-1, k]) / dy
        fluxes_y[i_glob, j_glob, k] = -Kh * grid.face_area_y[i_glob, j_glob, k] * dCdy
    end

    for k in axes(C_out, 3), j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_y[i_glob, j_glob+1, k] - fluxes_y[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
    end
end

get_grid_dims(grid::CartesianGrid) = Tuple(grid.dims)
get_grid_dims(grid::CurvilinearGrid) = (grid.nx, grid.ny, grid.nz)

get_dx_at_face(grid::CartesianGrid, i_phys, j_phys) = (grid.x[2+grid.ng,1+grid.ng,1] - grid.x[1+grid.ng,1+grid.ng,1])
get_dx_at_face(grid::CurvilinearGrid, i_phys, j_phys) = 1.0 / grid.pm[min(i_phys, grid.nx), j_phys]

get_dy_at_face(grid::CartesianGrid, i_phys, j_phys) = (grid.y[1+grid.ng,2+grid.ng,1] - grid.y[1+grid.ng,1+grid.ng,1])
get_dy_at_face(grid::CurvilinearGrid, i_phys, j_phys) = 1.0 / grid.pn[i_phys, min(j_phys, grid.ny)]

get_dx_centers(grid::CartesianGrid, i_phys, j_phys) = (grid.x[2+grid.ng,1+grid.ng,1] - grid.x[1+grid.ng,1+grid.ng,1])
get_dx_centers(grid::CurvilinearGrid, i_phys, j_phys) = 1 / (0.5 * (grid.pm[i_phys-1, j_phys] + grid.pm[i_phys, j_phys]))

get_dy_centers(grid::CartesianGrid, i_phys, j_phys) = (grid.y[1+grid.ng,2+grid.ng,1] - grid.y[1+grid.ng,1+grid.ng,1])
get_dy_centers(grid::CurvilinearGrid, i_phys, j_phys) = 1 / (0.5 * (grid.pn[i_phys, j_phys-1] + grid.pn[i_phys, j_phys]))

end # module HorizontalTransportModule

