module FluxLimitersModule

export calculate_limited_flux, van_leer, minmod, superbee, mc

"""
    van_leer(r::Float64)

Calculates phi based on van Leers method.
"""
function van_leer(r::Float64)
    return (r + abs(r)) / (1.0 + abs(r))
end

"""
    minmod(r::Float64)

Calculates phi based on the minmod limiter.
"""
function minmod(r::Float64)
    return max(0.0, min(1.0, r))
end

"""
    superbee(r::Float64)

Calculates phi based on the superbee limiter.
"""
function superbee(r::Float64)
    return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))
end

"""
    mc(r::Float64)

Calculates phi based on the Monotonized Central (MC) limiter.
"""
function mc(r::Float64)
    return max(0.0, min(2.0 * r, 0.5 * (1.0 + r), 2.0))
end


"""
    calculate_limited_flux(c_up_far::Float64, c_up_near::Float64, c_down_near::Float64, velocity::Float64, face_area::Float64, limiter_func::Function)

Calculates the advective flux across a cell face using a user-specified TVD flux limiter.

The function first determines the smoothness of the solution via the parameter `r`, which is the ratio of
consecutive gradients. This `r` value is then passed to the user-provided `limiter_func` to get a correction
factor `phi`. The final concentration at the cell face is a blend of a low-order (upwind) and a 
high-order
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
- `limiter_func`: A function (e.g., `van_leer`, `minmod`) that accepts `r::Float64` and returns `phi::Float64`.

# Returns
- `Float64`: The mass flux across the cell face.
"""
function calculate_limited_flux(c_up_far::Float64, c_up_near::Float64, c_down_near::Float64, velocity::Float64, face_area::Float64, limiter_func::Function)
    # Calculate the smoothness parameter 'r'
    numerator = c_up_near - c_up_far
    denominator = c_down_near - c_up_near
    
    r = if abs(denominator) < 1e-9
        # If denominator is zero, we are at an extremum or a flat region.
        # Returning 1.0 is a robust choice as most limiters
        # (van_leer, minmod, mc) evaluate to 1.0 at r=1.0.
        1.0
    else
        numerator / denominator
    end

    # Calculate the limiter function phi(r) using the provided function
    phi = limiter_func(r)

    # Calculate the limited concentration at the face
    # This is C_upwind + 0.5 * phi * (C_downwind - C_upwind)
    c_face_limited = c_up_near + 0.5 * phi * (c_down_near - c_up_near)

    # Calculate the final flux and return value
  return velocity * c_face_limited * face_area
end


end