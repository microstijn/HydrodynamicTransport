# src/State.jl

module StateModule

export initialize_state

using ..ModelStructs

"""
    initialize_state(grid::Grid, tracer_names::NTuple{N, Symbol} where N)

Create a `State` object with tracer fields initialized to zero.

This function allocates the memory for all dynamic fields in the simulation,
ensuring they are consistent with the dimensions of the provided `grid`.
Velocity fields are allocated with staggered dimensions appropriate for an
Arakawa C-grid.
"""
function initialize_state(grid::Grid, tracer_names::NTuple{N, Symbol} where N)
    # Get grid dimensions for convenience
    nx, ny, nz = grid.dims

    # Initialize the tracers dictionary (at cell centers)
    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx, ny, nz)
    end

    # Initialize hydrodynamic and environmental fields.
    # U, V, and W have staggered dimensions, with one extra element in their respective directions[cite: 1336].
    u = zeros(Float64, nx + 1, ny, nz) # Located on x-faces (between i-1 and i)
    v = zeros(Float64, nx, ny + 1, nz) # Located on y-faces (between j-1 and j)
    w = zeros(Float64, nx, ny, nz + 1) # Located on z-faces (between k-1 and k)

    # Scalar environmental fields remain at cell centers
    temperature = zeros(Float64, nx, ny, nz)
    salinity = zeros(Float64, nx, ny, nz)
    tss = zeros(Float64, nx, ny, nz)
    uvb = zeros(Float64, nx, ny, nz)

    # Construct and return the State object
    return State(tracers, u, v, w, temperature, salinity, tss, uvb)
end

end # module StateModule