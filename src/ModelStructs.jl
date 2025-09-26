# src/Structs.jl

module ModelStructs

export Grid, State

using StaticArrays

"""
    Grid

A struct to hold all the static grid information for the simulation.
This represents the Arakawa 'C' staggered grid.
"""
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

"""
    State

A struct to hold the dynamic variables of the simulation that change over time.
"""
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

end # module ModelStructs