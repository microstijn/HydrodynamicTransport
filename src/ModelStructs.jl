# src/ModelStructs.jl

module ModelStructs

export AbstractGrid, CartesianGrid, CurvilinearGrid, State, HydrodynamicData, PointSource

using StaticArrays
using Base: @kwdef

# Abstract type for all grid systems
"""
    AbstractGrid

An abstract supertype for all grid implementations. This allows functions
to be specialized for different grid types (e.g., Cartesian vs. Curvilinear)
through multiple dispatch.
"""
abstract type AbstractGrid end


# CartesianGrid struct
"""
    CartesianGrid <: AbstractGrid

A struct holding static geometric information for a uniform, rectangular computational domain.
"""
struct CartesianGrid <: AbstractGrid
    dims::SVector{3, Int}
    x::Array{Float64, 3}; y::Array{Float64, 3}; z::Array{Float64, 3}
    volume::Array{Float64, 3}
    face_area_x::Array{Float64, 3}; face_area_y::Array{Float64, 3}; face_area_z::Array{Float64, 3}
    mask::Array{Bool, 3}
end


# CurvilinearGrid struct
"""
    CurvilinearGrid <: AbstractGrid

A struct holding all static geometric and metric information for a curvilinear,
boundary-following grid, as read from a standard ocean model output file (e.g., ROMS).

# Fields
- `nx, ny, nz::Int`: Number of interior tracer grid cells.
- `lon_rho, lat_rho::Array{Float64, 2}`: Longitude/Latitude at tracer points.
- `lon_u, lat_u::Array{Float64, 2}`: Longitude/Latitude at U-velocity points.
- `lon_v, lat_v::Array{Float64, 2}`: Longitude/Latitude at V-velocity points.
- `z_w::Vector{Float64}`: Vertical coordinates at cell interfaces (w-points).
- `pm, pn::Array{Float64, 2}`: Inverse grid spacing (1/dx, 1/dy) at tracer points.
- `angle::Array{Float64, 2}`: Grid rotation angle at tracer points [radians].
- `h::Array{Float64, 2}`: Bathymetry at tracer points [m].
- `mask_rho, mask_u, mask_v::Array{Bool, 2}`: Land/Sea masks for each stagger location.
- `face_area_x, face_area_y::Array{Float64, 3}`: Area of faces normal to the grid axes.
- `volume::Array{Float64, 3}`: Volume of tracer cells.
"""
struct CurvilinearGrid <: AbstractGrid
    nx::Int; ny::Int; nz::Int
    lon_rho::Array{Float64, 2}; lat_rho::Array{Float64, 2}
    lon_u::Array{Float64, 2}; lat_u::Array{Float64, 2}
    lon_v::Array{Float64, 2}; lat_v::Array{Float64, 2}
    z_w::Vector{Float64}
    pm::Array{Float64, 2}; pn::Array{Float64, 2}
    angle::Array{Float64, 2}
    h::Array{Float64, 2}
    mask_rho::Array{Bool, 2}; mask_u::Array{Bool, 2}; mask_v::Array{Bool, 2}
    face_area_x::Array{Float64, 3}; face_area_y::Array{Float64, 3}
    volume::Array{Float64, 3}
end


# State, HydrodynamicData, and PointSources
struct State
    tracers::Dict{Symbol, Array{Float64, 3}}
    u::Array{Float64, 3}; v::Array{Float64, 3}; w::Array{Float64, 3}
    temperature::Array{Float64, 3}; salinity::Array{Float64, 3}
    tss::Array{Float64, 3}; uvb::Array{Float64, 3}
end

@kwdef struct PointSource
    i::Int; j::Int; k::Int
    tracer_name::Symbol
    influx_rate::Function
end

struct HydrodynamicData
    filepath::String
    var_map::Dict{Symbol, String}
end

end # module ModelStructs