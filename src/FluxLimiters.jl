module FluxLimiters

export calculate_limited_flux

"""
    van_leer(r::Float64)

Calculates phi based on van Leers method.
"""
function van_leer(r::Float64)
    return (r + abs(r)) / (1.0 + abs(r))
end

"""
    calculate_limited_flux(c_up_far::Float64, c_up_near::Float64, c_down_near::Float64, velocity::Float64, face_area::Float64)

Calculates the advective flux across a cell face using a TVD scheme with the van Leer flux limiter.

The function first determines the smoothness of the solution via the parameter `r`, which is the ratio of
consecutive gradients. This `r` value is then passed to the `van_leer` limiter function to get a correction
factor `phi`. The final concentration at the cell face is a blend of a low-order (upwind) and a high-order
(central difference) scheme, controlled by `phi`.

This approach ensures that in smooth regions of the flow, the scheme is high-order accurate, while near
sharp gradients or discontinuities, it reverts to a more stable, low-order scheme to prevent oscillations
and maintain monotonicity.

# Arguments
- `c_up_far`: Concentration in the cell "far" upstream.
- `c_up_near`: Concentration in the cell immediately upstream of the face (the donor cell).
- `c_down_near`: Concentration in the cell immediately downstream of the face (the receiver cell).
- `velocity`: The fluid velocity perpendicular to the face.
- `face_area`: The area of the cell face.

# Returns
- `Float64`: The mass flux across the cell face.
"""
function calculate_limited_flux(c_up_far::Float64, c_up_near::Float64, c_down_near::Float64, velocity::Float64, face_area::Float64)
    # 1. Calculate the smoothness parameter 'r'
    numerator = c_up_near - c_up_far
    denominator = c_down_near - c_up_near
    
    r = if abs(denominator) < 1e-9
        # If denominator is zero, we are at an extremum or a flat region.
        # Returning a large value or 1.0 is a common strategy. For van Leer,
        # r > 2 gives phi = 2, so any large number works.
        1.0
    else
        numerator / denominator
    end

    # 2. Calculate the limiter function phi(r)
    phi = van_leer(r)

    # 3. Calculate the limited concentration at the face
    # This is C_upwind + 0.5 * phi * (C_downwind - C_upwind)
    c_face_limited = c_up_near + 0.5 * phi * (c_down_near - c_up_near)

    # 4. Calculate the final flux
    return velocity * c_face_limited * face_area
end


end