# src/StateModule.jl

module StateModule

export initialize_state

using ..ModelStructs
using NCDatasets

# This is the internal helper for getting dimensions from our grid structs
_get_dims(grid::CartesianGrid) = grid.dims
_get_dims(grid::CurvilinearGrid) = (grid.nx, grid.ny, grid.nz)

"""
    initialize_state(grid::AbstractGrid, tracer_names)

Creates a `State` object for placeholder/test runs.
"""
function initialize_state(grid::AbstractGrid, tracer_names::NTuple{N, Symbol} where N)
    nx, ny, nz = _get_dims(grid)
    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx, ny, nz)
    end
    u = zeros(Float64, nx + 1, ny, nz)
    v = zeros(Float64, nx, ny + 1, nz)
    w = zeros(Float64, nx, ny, nz + 1)
    temperature = zeros(Float64, nx, ny, nz)
    salinity = zeros(Float64, nx, ny, nz)
    tss = zeros(Float64, nx, ny, nz)
    uvb = zeros(Float64, nx, ny, nz)
    return State(tracers, u, v, w, temperature, salinity, tss, uvb)
end

"""
    initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names)

Creates a `State` object with dimensions that perfectly match the NetCDF file `ds`.
This is the robust method for real data runs.
"""
function initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names::NTuple{N, Symbol} where N)
    # Get tracer dimensions from the grid struct
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx, ny, nz)
    end

    # --- FIX: Get the EXACT staggered dimensions from the NetCDF file ---
    u_dims = (ds.dim["xi_u"], ds.dim["eta_u"], ds.dim["s_rho"])
    v_dims = (ds.dim["xi_v"], ds.dim["eta_v"], ds.dim["s_rho"])
    w_dims = (ds.dim["xi_rho"], ds.dim["eta_rho"], ds.dim["s_w"])

    u = zeros(Float64, u_dims)
    v = zeros(Float64, v_dims)
    w = zeros(Float64, w_dims)

    # Scalar fields use the tracer dimensions
    temperature = zeros(Float64, nx, ny, nz)
    salinity = zeros(Float64, nx, ny, nz)
    tss = zeros(Float64, nx, ny, nz)
    uvb = zeros(Float64, nx, ny, nz)

    return State(tracers, u, v, w, temperature, salinity, tss, uvb)
end


end # module StateModule