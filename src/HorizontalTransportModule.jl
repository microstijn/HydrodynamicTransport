# src/HorizontalTransportModule.jl

module HorizontalTransportModule

export horizontal_transport!

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.BoundaryConditionsModule: apply_intermediate_boundary_conditions!
using StaticArrays
using Logging
using LinearAlgebra


"""
    horizontal_transport!(state::State, grid::AbstractGrid, dt::Float64, scheme::Symbol, D_crit::Float64, boundary_conditions::Vector{<:BoundaryCondition})

Computes the change in tracer concentrations due to horizontal transport processes
(advection and diffusion) over a single time step.

This function acts as the main dispatcher for horizontal transport. It uses operator
splitting to solve the transport equation.

The specific advection algorithm is chosen via the `scheme` argument.

# Arguments
- `state::State`: The model state, which is modified in-place.
- `grid::AbstractGrid`: The computational grid.
- `dt::Float64`: The time step duration.
- `scheme::Symbol`: The advection scheme to use. Options are `:TVD`, `:UP3`, and `:ImplicitADI`.
- `D_crit::Float64`: Critical depth for wet/dry cells.
- `boundary_conditions::Vector{<:BoundaryCondition}`: A vector of boundary conditions.

# Returns
- `nothing`: The function modifies `state.tracers` in-place.
"""
function horizontal_transport!(state::State, grid::AbstractGrid, dt::Float64, scheme::Symbol, D_crit::Float64, boundary_conditions::Vector{<:BoundaryCondition})
    Kh = 1.0 
    for tracer_name in keys(state.tracers)
        C_initial = state.tracers[tracer_name]
        C_intermediate = state._buffers[tracer_name] # Re-using buffer
        
        # --- Advection Step ---
        if scheme == :TVD
            advect_x_tvd!(C_intermediate, C_initial, state, grid, dt, state.flux_x, D_crit)
            advect_y_tvd!(C_initial, C_intermediate, state, grid, dt, state.flux_y, D_crit)
        elseif scheme == :UP3
            advect_x_up3!(C_intermediate, C_initial, state, grid, dt, state.flux_x, D_crit)
            advect_y_up3!(C_initial, C_intermediate, state, grid, dt, state.flux_y, D_crit)
        elseif scheme == :ImplicitADI
            # Step 1: Implicit X-Sweep: (I - dt*L_x) * C_intermediate = C_initial
            advect_implicit_x!(C_intermediate, C_initial, state, grid, dt)

            # Step 1.5: Handle Boundary Conditions for the Intermediate Variable
            apply_intermediate_boundary_conditions!(C_intermediate, C_initial, grid, boundary_conditions, tracer_name)

            # Step 2: Implicit Y-Sweep: (I - dt*L_y) * C_final = C_intermediate
            advect_implicit_y!(C_initial, C_intermediate, state, grid, dt) # Result stored back in C_initial
        else
            error("Unknown advection scheme: $scheme. Available options are :TVD, :UP3, and :ImplicitADI.")
        end
        
        # --- Diffusion Step (common to all schemes) ---
        # The result of advection is in C_initial (now C_final_advection)
        # We use C_intermediate as the buffer again.
        diffuse_x!(C_intermediate, C_initial, state, grid, dt, Kh, state.flux_x, D_crit)
        diffuse_y!(C_initial, C_intermediate, state, grid, dt, Kh, state.flux_y, D_crit)
    end
    return nothing
end


# --- Stencil Functions (inlined for performance) ---
@inline function get_stencil_x(C::Array{Float64, 3}, i_glob::Int, j_glob::Int, k::Int)
    @inbounds return C[i_glob-2, j_glob, k], C[i_glob-1, j_glob, k], C[i_glob, j_glob, k], C[i_glob+1, j_glob, k], C[i_glob+2, j_glob, k]
end

@inline function get_stencil_y(C::Array{Float64, 3}, i_glob::Int, j_glob::Int, k::Int)
    @inbounds return C[i_glob, j_glob-2, k], C[i_glob, j_glob-1, k], C[i_glob, j_glob, k], C[i_glob, j_glob+1, k], C[i_glob, j_glob+2, k]
end


# ==============================================================================
# --- SCHEME 1: Upstream-Biased 3rd-Order (UP3) ---
# ==============================================================================

function advect_x_up3!(C_out, C_in, state::State, grid::AbstractGrid, dt, fluxes_x, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    u = state.u
    fluxes_x .= 0.0 # Clear the buffer

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_in, 3)
        j_glob = j_phys + ng
        # --- Interior Faces (3rd-Order Upstream) ---
        for i_phys in 2:nx
            i_glob = i_phys + ng
            velocity = u[i_glob, j_glob, k]
            
            # --- Cell-Face Blocking Logic ---
            if isa(grid, CurvilinearGrid)
                if velocity > 0 # Flow is from left to right, check the left cell
                    upstream_depth = grid.h[i_glob-1, j_glob] + state.zeta[i_glob-1, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                elseif velocity < 0 # Flow is from right to left, check the right cell
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                end
            end
            # --- End Cell-Face Blocking ---

            local C_face
            if velocity >= 0 # Flow is from left to right
                C_face = (2*C_in[i_glob, j_glob, k] + 5*C_in[i_glob-1, j_glob, k] - C_in[i_glob-2, j_glob, k]) / 6
            else # Flow is from right to left
                C_face = (2*C_in[i_glob-1, j_glob, k] + 5*C_in[i_glob, j_glob, k] - C_in[i_glob+1, j_glob, k]) / 6
            end
            fluxes_x[i_glob, j_glob, k] = velocity * C_face * grid.face_area_x[i_glob, j_glob, k]
        end

        # --- Boundary Faces (1st-Order Upwind Fallback) ---
        for i_phys in [1, nx+1]
            i_glob = i_phys + ng
            vel = u[i_glob, j_glob, k]

            # --- Cell-Face Blocking Logic ---
            if isa(grid, CurvilinearGrid)
                if vel > 0 # Flow L->R, upstream cell is i_glob-1
                    upstream_depth = grid.h[i_glob-1, j_glob] + state.zeta[i_glob-1, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                elseif vel < 0 # Flow R->L, upstream cell is i_glob
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                end
            end
            # --- End Cell-Face Blocking ---

            C_face = (vel >= 0) ? C_in[i_glob-1, j_glob, k] : C_in[i_glob, j_glob, k]
            fluxes_x[i_glob, j_glob, k] = vel * C_face * grid.face_area_x[i_glob, j_glob, k]
        end
      end
    end

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_x[i_glob+1, j_glob, k] - fluxes_x[i_glob, j_glob, k]
        if grid.volume[i_glob, j_glob, k] > 0
             C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
        else
             C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k]
        end
      end
    end
end

function advect_y_up3!(C_out, C_in, state::State, grid::AbstractGrid, dt, fluxes_y, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    v = state.v
    fluxes_y .= 0.0 # Clear the buffer

    @inbounds Base.Threads.@threads for i_phys in 1:nx
      for k in axes(C_in, 3)
        i_glob = i_phys + ng
        # --- Interior Faces (3rd-Order Upstream) ---
        for j_phys in 2:ny
            j_glob = j_phys + ng
            velocity = v[i_glob, j_glob, k]
            
            # --- Cell-Face Blocking Logic ---
            if isa(grid, CurvilinearGrid)
                if velocity > 0 # Flow is from bottom to top, check bottom cell
                    upstream_depth = grid.h[i_glob, j_glob-1] + state.zeta[i_glob, j_glob-1, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                elseif velocity < 0 # Flow is from top to bottom, check top cell
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                end
            end
            # --- End Cell-Face Blocking ---

            local C_face
            if velocity >= 0 # Flow is from bottom to top
                C_face = (2*C_in[i_glob, j_glob, k] + 5*C_in[i_glob, j_glob-1, k] - C_in[i_glob, j_glob-2, k]) / 6
            else # Flow is from top to bottom
                C_face = (2*C_in[i_glob, j_glob-1, k] + 5*C_in[i_glob, j_glob, k] - C_in[i_glob, j_glob+1, k]) / 6
            end
            fluxes_y[i_glob, j_glob, k] = velocity * C_face * grid.face_area_y[i_glob, j_glob, k]
        end

        # --- Boundary Faces (1st-Order Upwind Fallback) ---
        for j_phys in [1, ny+1]
            j_glob = j_phys + ng
            vel = v[i_glob, j_glob, k]
            
            # --- Cell-Face Blocking Logic ---
            if isa(grid, CurvilinearGrid)
                if vel > 0 # Flow bottom->top, upstream cell is j_glob-1
                    upstream_depth = grid.h[i_glob, j_glob-1] + state.zeta[i_glob, j_glob-1, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                elseif vel < 0 # Flow top->bottom, upstream cell is j_glob
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                end
            end
            # --- End Cell-Face Blocking ---

            C_face = (vel >= 0) ? C_in[i_glob, j_glob-1, k] : C_in[i_glob, j_glob, k]
            fluxes_y[i_glob, j_glob, k] = vel * C_face * grid.face_area_y[i_glob, j_glob, k]
        end
      end
    end
    
    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_y[i_glob, j_glob+1, k] - fluxes_y[i_glob, j_glob, k]
        if grid.volume[i_glob, j_glob, k] > 0
            C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
        else
            C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k]
        end
      end
    end
end


# ==============================================================================
# --- SCHEME 2: Total Variation Diminishing (TVD) based on Bott Scheme ---
# ==============================================================================

"""
    calculate_bott_coeffs(c_im2, c_im1, c_i, c_ip1, c_ip2)

Calculates the coefficients of a 4th-order polynomial that interpolates the
concentration profile over a 5-point stencil. This is a core component of the
Bott (1989) advection scheme.

# Arguments
- `c_im2`, `c_im1`, `c_i`, `c_ip1`, `c_ip2`: Concentration values at five consecutive
  grid points (i-2, i-1, i, i+1, i+2).

# Returns
- `(a0, a1, a2, a3, a4)`: A tuple of the five polynomial coefficients.
"""
@inline function calculate_bott_coeffs(c_im2, c_im1, c_i, c_ip1, c_ip2)
    a1 = (c_ip1 - c_im1) / 2.0
    a3 = (c_ip2 - 2*c_ip1 + 2*c_im1 - c_im2) / 12.0 - (2/3.0) * a1
    a4 = (c_ip2 - 4*c_ip1 + 6*c_i - 4*c_im1 + c_im2) / 24.0
    a0 = c_i - a4; a2 = a4
    return a0, a1, a2, a3, a4
end

"""
    _indefinite_integral_poly4(xi, a0, a1, a2, a3, a4)

Analytically computes the indefinite integral of the 4th-order polynomial defined
by the given coefficients. The integration is performed with respect to the normalized
coordinate `xi`.

# Arguments
- `xi`: The normalized coordinate (from -0.5 to 0.5) at which to evaluate the integral.
- `a0` to `a4`: The coefficients of the polynomial.

# Returns
- `Float64`: The value of the indefinite integral at `xi`.
"""
@inline function _indefinite_integral_poly4(xi, a0, a1, a2, a3, a4)
    return xi * (a0 + xi/2 * (a1 + xi/3 * (a2 + xi/4 * (a3 + xi/5 * a4))))
end


"""
    advect_x_tvd!(C_out, C_in, u, grid, dt)

Performs advection in the x-direction using a Total Variation Diminishing (TVD)
scheme based on the work of Bott (1989).

This function calculates advective fluxes across x-faces. For the deep interior of
the domain, it employs a high-order method:
1.  A 4th-order polynomial is fitted to a 5-point stencil around the donor cell.
2.  This polynomial is analytically integrated over the volume of fluid that crosses
    the cell face during the time step to calculate a high-order flux.
3.  A flux limiter (`phi`) is calculated to blend the high-order flux with a
    low-order (1st-order upwind) flux. This ensures that the scheme is TVD,
    preventing the formation of new, unphysical oscillations.
4.  The final flux is constrained to be between the concentrations of the donor
    and receiver cells to maintain monotonicity.

For faces near the domain boundaries, the function reverts to a simple and robust
1st-order upwind scheme.

# Arguments
- `C_out`: The output concentration array (modified in-place).
- `C_in`: The input concentration array.
- `u`: The u-component of velocity.
- `grid`: The computational grid.
- `dt`: The time step.

# Returns
- `nothing`: Modifies `C_out` in-place.
"""
function advect_x_tvd!(C_out, C_in, state::State, grid::AbstractGrid, dt, fluxes_x, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    u = state.u
    fluxes_x .= 0.0

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_in, 3)
        j_glob = j_phys + ng
        # --- Interior Faces ---
        for i_phys in 3:nx-1
            i_glob = i_phys + ng
            velocity = u[i_glob, j_glob, k]
            if abs(velocity) < 1e-12; continue; end

            if isa(grid, CurvilinearGrid)
                if velocity > 0 # Flow L->R
                    upstream_depth = grid.h[i_glob-1, j_glob] + state.zeta[i_glob-1, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                else # Flow R->L
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                end
            end

            donor_idx, receiver_idx = velocity >= 0 ? (i_glob - 1, i_glob) : (i_glob, i_glob - 1)
            c_stencil = get_stencil_x(C_in, donor_idx, j_glob, k)
            dx_donor = get_dx_at_face(grid, i_glob, j_glob)
            a0,a1,a2,a3,a4 = calculate_bott_coeffs(c_stencil...)
            courant_abs = abs(velocity * dt / dx_donor)
            if courant_abs > 1.0; courant_abs = 1.0; end
            
            integral_right, integral_left = if velocity >= 0
                _indefinite_integral_poly4(0.5, a0,a1,a2,a3,a4), _indefinite_integral_poly4(0.5-courant_abs, a0,a1,a2,a3,a4)
            else
                _indefinite_integral_poly4(-0.5+courant_abs, a0,a1,a2,a3,a4), _indefinite_integral_poly4(-0.5, a0,a1,a2,a3,a4)
            end
            
            C_face_high_order = (integral_right - integral_left) / (courant_abs + 1e-12)
            C_face_low_order = C_in[donor_idx, j_glob, k]
            c_up_far=C_in[donor_idx - (velocity >= 0 ? 1 : -1), j_glob, k]
            c_up_near=C_in[donor_idx, j_glob, k]
            c_down_near=C_in[receiver_idx, j_glob, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
            r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator
            phi = (r + abs(r)) / (1 + abs(r) + 1e-9)
            C_face_tvd = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)
            C_max = max(C_in[donor_idx, j_glob, k], C_in[receiver_idx, j_glob, k])
            C_min = min(C_in[donor_idx, j_glob, k], C_in[receiver_idx, j_glob, k])
            fluxes_x[i_glob, j_glob, k] = velocity * max(C_min, min(C_max, C_face_tvd)) * grid.face_area_x[i_glob, j_glob, k]
        end
        
        # --- Boundary Faces ---
        for i_phys in [1, 2, nx, nx+1]
            i_glob = i_phys + ng
            vel = u[i_glob, j_glob, k]
            if isa(grid, CurvilinearGrid)
                if vel > 0
                    upstream_depth = grid.h[i_glob-1, j_glob] + state.zeta[i_glob-1, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                elseif vel < 0
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                end
            end
            fluxes_x[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob-1, j_glob, k] : C_in[i_glob, j_glob, k]) * grid.face_area_x[i_glob, j_glob, k]
        end
      end
    end

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_x[i_glob+1, j_glob, k] - fluxes_x[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
      end
    end
end

function advect_y_tvd!(C_out, C_in, state::State, grid::AbstractGrid, dt, fluxes_y, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    v = state.v
    fluxes_y .= 0.0

    @inbounds Base.Threads.@threads for i_phys in 1:nx
      for k in axes(C_in, 3)
        i_glob = i_phys + ng
        # --- Interior Faces ---
        for j_phys in 3:ny-1
            j_glob = j_phys + ng
            velocity = v[i_glob, j_glob, k]
            if abs(velocity) < 1e-12; continue; end

            if isa(grid, CurvilinearGrid)
                if velocity > 0 # Flow bottom->top
                    upstream_depth = grid.h[i_glob, j_glob-1] + state.zeta[i_glob, j_glob-1, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                else # Flow top->bottom
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; velocity = 0.0; end
                end
            end

            donor_idx, receiver_idx = velocity >= 0 ? (j_glob - 1, j_glob) : (j_glob, j_glob - 1)
            c_stencil = get_stencil_y(C_in, i_glob, donor_idx, k)
            dy_donor = get_dy_at_face(grid, i_glob, j_glob)
            a0,a1,a2,a3,a4 = calculate_bott_coeffs(c_stencil...)
            courant_abs = abs(velocity * dt / dy_donor)
            if courant_abs > 1.0; courant_abs = 1.0; end
            
            integral_right, integral_left = if velocity >= 0
                _indefinite_integral_poly4(0.5, a0,a1,a2,a3,a4), _indefinite_integral_poly4(0.5-courant_abs, a0,a1,a2,a3,a4)
            else
                _indefinite_integral_poly4(-0.5+courant_abs, a0,a1,a2,a3,a4), _indefinite_integral_poly4(-0.5, a0,a1,a2,a3,a4)
            end

            C_face_high_order = (integral_right - integral_left) / (courant_abs + 1e-12)
            C_face_low_order = C_in[i_glob, donor_idx, k]
            c_up_far=C_in[i_glob, donor_idx - (velocity >= 0 ? 1 : -1), k]
            c_up_near=C_in[i_glob, donor_idx, k]
            c_down_near=C_in[i_glob, receiver_idx, k]
            r_numerator = c_up_near - c_up_far; r_denominator = c_down_near - c_up_near
            r = abs(r_denominator) < 1e-9 ? 1.0 : r_numerator / r_denominator
            phi = (r + abs(r)) / (1 + abs(r) + 1e-9)
            C_face_tvd = C_face_low_order + 0.5 * phi * (C_face_high_order - C_face_low_order)
            C_max = max(C_in[i_glob, donor_idx, k], C_in[i_glob, receiver_idx, k])
            C_min = min(C_in[i_glob, donor_idx, k], C_in[i_glob, receiver_idx, k])
            fluxes_y[i_glob, j_glob, k] = velocity * max(C_min, min(C_max, C_face_tvd)) * grid.face_area_y[i_glob, j_glob, k]
        end

        # --- Boundary Faces ---
        for j_phys in [1, 2, ny, ny+1]
            j_glob = j_phys + ng
            vel = v[i_glob, j_glob, k]
            if isa(grid, CurvilinearGrid)
                if vel > 0
                    upstream_depth = grid.h[i_glob, j_glob-1] + state.zeta[i_glob, j_glob-1, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                elseif vel < 0
                    upstream_depth = grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, k]
                    if upstream_depth < D_crit; vel = 0.0; end
                end
            end
            fluxes_y[i_glob, j_glob, k] = vel * (vel >= 0 ? C_in[i_glob, j_glob-1, k] : C_in[i_glob, j_glob, k]) * grid.face_area_y[i_glob, j_glob, k]
        end
      end
    end

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_y[i_glob, j_glob+1, k] - fluxes_y[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
      end
    end
end

# ==============================================================================
# --- SCHEME 3: Implicit Advection (ADI) ---
# ==============================================================================

"""
    advect_implicit_x!(C_intermediate, C_initial, state, grid, dt)

Performs the first step of the ADI sequence (implicit x-sweep).
Solves `(I - dt*L_x) * C_intermediate = C_initial` for each row.
"""
function advect_implicit_x!(C_intermediate::Array{Float64, 3}, C_initial::Array{Float64, 3}, state::State, grid::AbstractGrid, dt::Float64)
    nx, ny, nz = get_grid_dims(grid)
    ng = grid.ng
    u = state.u

    # Pre-allocate vectors for the tridiagonal system
    a = Vector{Float64}(undef, nx - 1) # sub-diagonal
    b = Vector{Float64}(undef, nx)     # main diagonal
    c = Vector{Float64}(undef, nx - 1) # super-diagonal
    d = Vector{Float64}(undef, nx)     # RHS

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in 1:nz
        j_glob = j_phys + ng
        
        # --- 1. Construct the tridiagonal system for the current row ---
        for i_phys in 1:nx
            i_glob = i_phys + ng
            
            # Enforce zero-flux boundary conditions for the solver
            u_left = (i_phys == 1) ? 0.0 : u[i_glob, j_glob, k]
            u_right = (i_phys == nx) ? 0.0 : u[i_glob + 1, j_glob, k]

            dx_i = get_dx_at_face(grid, i_glob, j_glob)
            dx_ip1 = get_dx_at_face(grid, i_glob + 1, j_glob)
            cr_left = (dt / dx_i) * u_left
            cr_right = (dt / dx_ip1) * u_right
            
            alpha = max(cr_left, 0)
            gamma = min(cr_right, 0)
            beta = max(cr_right, 0) - min(cr_left, 0)

            if i_phys > 1; a[i_phys-1] = -alpha; end
            b[i_phys] = 1 + beta
            if i_phys < nx; c[i_phys] = gamma; end
            d[i_phys] = C_initial[i_glob, j_glob, k]
        end

        # --- 2. Solve the system using LinearAlgebra ---
        A = Tridiagonal(a, b, c)
        solution = A \ d
        
        # --- 3. Store the result in the intermediate buffer's physical domain ---
        view(C_intermediate, (ng+1):(nx+ng), j_glob, k) .= solution
      end
    end
end

"""
    advect_implicit_y!(C_final, C_intermediate, state, grid, dt)

Performs the second step of the ADI sequence (implicit y-sweep).
Solves `(I - dt*L_y) * C_final = C_intermediate` for each column.
"""
function advect_implicit_y!(C_final::Array{Float64, 3}, C_intermediate::Array{Float64, 3}, state::State, grid::AbstractGrid, dt::Float64)
    nx, ny, nz = get_grid_dims(grid)
    ng = grid.ng
    v = state.v

    # Pre-allocate vectors for the tridiagonal system
    a = Vector{Float64}(undef, ny - 1) # sub-diagonal
    b = Vector{Float64}(undef, ny)     # main-diagonal
    c = Vector{Float64}(undef, ny - 1) # super-diagonal
    d = Vector{Float64}(undef, ny)     # RHS

    @inbounds Base.Threads.@threads for i_phys in 1:nx
      for k in 1:nz
        i_glob = i_phys + ng

        # --- 1. Construct the tridiagonal system for the current column ---
        for j_phys in 1:ny
            j_glob = j_phys + ng
            
            # Enforce zero-flux boundary conditions for the solver
            v_bottom = (j_phys == 1) ? 0.0 : v[i_glob, j_glob, k]
            v_top = (j_phys == ny) ? 0.0 : v[i_glob, j_glob + 1, k]

            dy_j = get_dy_at_face(grid, i_glob, j_glob)
            dy_jp1 = get_dy_at_face(grid, i_glob, j_glob + 1)
            cr_bottom = (dt / dy_j) * v_bottom
            cr_top = (dt / dy_jp1) * v_top

            alpha = max(cr_bottom, 0)
            gamma = min(cr_top, 0)
            beta = max(cr_top, 0) - min(cr_bottom, 0)

            if j_phys > 1; a[j_phys-1] = -alpha; end
            b[j_phys] = 1 + beta
            if j_phys < ny; c[j_phys] = gamma; end
            d[j_phys] = C_intermediate[i_glob, j_glob, k]
        end

        # --- 2. Solve the system using LinearAlgebra ---
        A = Tridiagonal(a, b, c)
        solution = A \ d

        # --- 3. Store the result back into the final tracer array's physical domain ---
        view(C_final, i_glob, (ng+1):(ny+ng), k) .= solution
      end
    end
end


# ==============================================================================
# --- Diffusion and Helper Functions ---
# ==============================================================================

"""
    diffuse_x!(C_out, C_in, grid, dt, Kh)

Calculates the change in concentration due to horizontal diffusion in the x-direction.

This function computes diffusive fluxes across the x-faces of the grid cells based on
the concentration gradient and the horizontal diffusivity coefficient `Kh`. It updates
the `C_out` array based on the divergence of these fluxes. A no-flux condition is
enforced at land boundaries by checking the grid mask.

# Arguments
- `C_out`: The output concentration array (modified in-place).
- `C_in`: The input concentration array.
- `grid`: The computational grid.
- `dt`: The time step.
- `Kh`: The horizontal diffusion coefficient.

# Returns
- `nothing`: Modifies `C_out` in-place.
"""
function diffuse_x!(C_out, C_in, state::State, grid::AbstractGrid, dt, Kh, fluxes_x, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    fluxes_x .= 0.0
    
    # Calculate fluxes only for interior faces, enforcing zero-flux at boundaries.
    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_in, 3), i_phys in 2:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        local flux = 0.0
        # --- Cell-Face Blocking Logic for Diffusion ---
        if isa(grid, CurvilinearGrid)
            depth1 = grid.h[i_glob-1, j_glob] + state.zeta[i_glob-1, j_glob, k]
            depth2 = grid.h[i_glob, j_glob]   + state.zeta[i_glob, j_glob, k]
            if depth1 < D_crit || depth2 < D_crit
                flux = 0.0
            else
                dx = get_dx_centers(grid, i_glob, j_glob)
                dCdx = (C_in[i_glob, j_glob, k] - C_in[i_glob-1, j_glob, k]) / dx
                flux = -Kh * grid.face_area_x[i_glob, j_glob, k] * dCdx
            end
        else # Original logic for CartesianGrid or when not using blocking
            dx = get_dx_centers(grid, i_glob, j_glob)
            dCdx = (C_in[i_glob, j_glob, k] - C_in[i_glob-1, j_glob, k]) / dx
            flux = -Kh * grid.face_area_x[i_glob, j_glob, k] * dCdx
        end

        face_is_wet = isa(grid, CurvilinearGrid) ? grid.mask_u[i_glob, j_glob] : (grid.mask[i_glob, j_glob, k] & grid.mask[i_glob-1, j_glob, k])
        fluxes_x[i_glob, j_glob, k] = flux * face_is_wet
      end
    end

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_x[i_glob+1, j_glob, k] - fluxes_x[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
      end
    end
end

function diffuse_y!(C_out, C_in, state::State, grid::AbstractGrid, dt, Kh, fluxes_y, D_crit::Float64)
    nx, ny, _ = get_grid_dims(grid)
    ng = grid.ng
    fluxes_y .= 0.0
    
    # Calculate fluxes only for interior faces, enforcing zero-flux at boundaries.
    @inbounds Base.Threads.@threads for i_phys in 1:nx
      for k in axes(C_in, 3), j_phys in 2:ny
        i_glob, j_glob = i_phys + ng, j_phys + ng
        
        local flux = 0.0
        # --- Cell-Face Blocking Logic for Diffusion ---
        if isa(grid, CurvilinearGrid)
            depth1 = grid.h[i_glob, j_glob-1] + state.zeta[i_glob, j_glob-1, k]
            depth2 = grid.h[i_glob, j_glob]   + state.zeta[i_glob, j_glob, k]
            if depth1 < D_crit || depth2 < D_crit
                flux = 0.0
            else
                dy = get_dy_centers(grid, i_glob, j_glob)
                dCdy = (C_in[i_glob, j_glob, k] - C_in[i_glob, j_glob-1, k]) / dy
                flux = -Kh * grid.face_area_y[i_glob, j_glob, k] * dCdy
            end
        else # Original logic for CartesianGrid or when not using blocking
            dy = get_dy_centers(grid, i_glob, j_glob)
            dCdy = (C_in[i_glob, j_glob, k] - C_in[i_glob, j_glob-1, k]) / dy
            flux = -Kh * grid.face_area_y[i_glob, j_glob, k] * dCdy
        end

        face_is_wet = isa(grid, CurvilinearGrid) ? grid.mask_v[i_glob, j_glob] : (grid.mask[i_glob, j_glob, k] & grid.mask[i_glob, j_glob-1, k])
        fluxes_y[i_glob, j_glob, k] = flux * face_is_wet
      end
    end

    @inbounds Base.Threads.@threads for j_phys in 1:ny
      for k in axes(C_out, 3), i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        flux_divergence = fluxes_y[i_glob, j_glob+1, k] - fluxes_y[i_glob, j_glob, k]
        C_out[i_glob, j_glob, k] = C_in[i_glob, j_glob, k] - (dt / grid.volume[i_glob, j_glob, k]) * flux_divergence
      end
    end
end

@inline get_grid_dims(grid::CartesianGrid) = Tuple(grid.dims)
@inline get_grid_dims(grid::CurvilinearGrid) = (grid.nx, grid.ny, grid.nz)
@inline get_dx_at_face(grid::CartesianGrid, i_glob, j_glob) = (grid.x[2+grid.ng,1+grid.ng,1] - grid.x[1+grid.ng,1+grid.ng,1])
@inline get_dx_at_face(grid::CurvilinearGrid, i_glob, j_glob) = 1.0 / grid.pm[i_glob, j_glob]
@inline get_dy_at_face(grid::CartesianGrid, i_glob, j_glob) = (grid.y[1+grid.ng,2+grid.ng,1] - grid.y[1+grid.ng,1+grid.ng,1])
@inline get_dy_at_face(grid::CurvilinearGrid, i_glob, j_glob) = 1.0 / grid.pn[i_glob, j_glob]
@inline get_dx_centers(grid::CartesianGrid, i_glob, j_glob) = (grid.x[2+grid.ng,1+grid.ng,1] - grid.x[1+grid.ng,1+grid.ng,1])
@inline get_dx_centers(grid::CurvilinearGrid, i_glob, j_glob) = 1 / (0.5 * (grid.pm[i_glob-1, j_glob] + grid.pm[i_glob, j_glob]))
@inline get_dy_centers(grid::CartesianGrid, i_glob, j_glob) = (grid.y[1+grid.ng,2+grid.ng,1] - grid.y[1+grid.ng,1+grid.ng,1])
@inline get_dy_centers(grid::CurvilinearGrid, i_glob, j_glob) = 1 / (0.5 * (grid.pn[i_glob, j_glob-1] + grid.pn[i_glob, j_glob]))

end # module HorizontalTransportModule
