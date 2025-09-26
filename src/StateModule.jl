# src/State.jl

module StateModule

export initialize_state

using ..ModelStructs

"""
    initialize_state(grid::Grid, tracer_names::NTuple{N, Symbol} where N)

  Create a `State` object with tracer fields initialized to zero.

  This function allocates the memory for all dynamic fields in the simulation,
  ensuring they are consistent with the dimensions of the provided `grid`.
  Hydrodynamic and environmental fields are initialized as zero-filled placeholders,
  intended to be populated later by reading from input files.

  # Arguments
  - `grid::Grid`: A `Grid` object, created by `initialize_grid`.
  - `tracer_names`: A tuple of `Symbol`s specifying the names of the tracers to be
    included in the simulation (e.g., `(:C_dissolved, :C_sorbed)`).

  # Returns
  - A `State` object with all fields initialized to zero arrays of the correct size.

  # Examples
  ```jldoctest
  julia> using HydrodynamicTransport

  julia> grid = initialize_grid(10, 20, 5, 100.0, 200.0, 50.0);

  julia> tracer_names = (:virus_free, :virus_sorbed);

  julia> state = initialize_state(grid, tracer_names);

  julia> size(state.u)
  (10, 20, 5)

  julia> state.tracers[:virus_free][5, 10, 3]
  0.0
"""
function initialize_state(grid::Grid, tracer_names::NTuple{N, Symbol} where N)

  # Get grid dimensions for convenience
  dims = grid.dims

  # Initialize the tracers dictionary
  tracers = Dict{Symbol, Array{Float64, 3}}()
  for name in tracer_names
    tracers[name] = zeros(Float64, dims...)
  end

  # Initialize hydrodynamic and environmental fields as placeholders
  u = zeros(Float64, dims...)
  v = zeros(Float64, dims...)
  w = zeros(Float64, dims...)
  temperature = zeros(Float64, dims...)
  salinity = zeros(Float64, dims...)
  tss = zeros(Float64, dims...)
  uvb = zeros(Float64, dims...)

  # Construct and return the State object
  return State(tracers, u, v, w, temperature, salinity, tss, uvb)
end

end # module StateModule
