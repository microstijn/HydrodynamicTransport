# src/Structs.jl

module ModelStructs

export Grid, State, HydrodynamicData, PointSource

using StaticArrays
using Base: @kwdef # Import the macro

# Grid struct remains the same...
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

# State struct remains the same...
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

# HydrodynamicData struct remains the same...
struct HydrodynamicData
    filepath::String
end

# --- FIX: Add @kwdef to enable keyword arguments ---
@kwdef struct PointSource
    i::Int
    j::Int
    k::Int
    tracer_name::Symbol
    influx_rate::Function
end


end # module ModelStructs