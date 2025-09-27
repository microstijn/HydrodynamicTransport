# src/Structs.jl

module ModelStructs

export Grid, State, HydrodynamicData, PointSource

using StaticArrays
using Base: @kwdef # Import the macro

struct Grid
    dims::SVector{3, Int}
    x::Array{Float64, 3}
    y::Array{Float64, 3}
    z::Array{Float64, 3}
    volume::Array{Float64, 3}
    face_area_x::Array{Float64, 3}
    face_area_y::Array{Float64, 3}
    face_area_z::Array{Float64, 3}
    mask::Array{Bool, 3}
end

struct State
    tracers::Dict{Symbol, Array{Float64, 3}}
    u::Array{Float64, 3}
    v::Array{Float64, 3}
    w::Array{Float64, 3}
    temperature::Array{Float64, 3}
    salinity::Array{Float64, 3}
    tss::Array{Float64, 3}
    uvb::Array{Float64, 3}
end

"""
    HydrodynamicData

A configuration struct that holds information about the external hydrodynamic data source.

# Fields
- `filepath::String`: The path or URL to the data file.
- `var_map::Dict{Symbol, String}`: A dictionary that maps the model's internal, standardized
  variable names (e.g., `:u`, `:v`, `:time`) to the specific variable names used
  in the NetCDF file (e.g., `"water_u"`, `"water_v"`, `"ocean_time"`).
"""
struct HydrodynamicData
    filepath::String
    var_map::Dict{Symbol, String}
end

# Added @kwdef to enable keyword arguments
@kwdef struct PointSource
    i::Int
    j::Int
    k::Int
    tracer_name::Symbol
    influx_rate::Function
end


end # module ModelStructs