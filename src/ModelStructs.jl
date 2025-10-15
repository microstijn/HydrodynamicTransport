# src/ModelStructs.jl

module ModelStructs

export AbstractGrid, CartesianGrid, CurvilinearGrid, State, SedimentParams, HydrodynamicData, PointSource, 
       BoundaryCondition, OpenBoundary, RiverBoundary, TidalBoundary, FunctionalInteraction

using StaticArrays
using Base: @kwdef

@kwdef struct SedimentParams
    rho_fluid::Float64 = 1025.0
    rho_particle::Float64 = 2650.0
    manning_n::Float64 = 0.025
    ws0::Float64 = 0.001
    n_exponent::Float64 = 4.65
    tau_d::Float64 = 0.1
    tau_cr::Float64 = 0.2
    M::Float64 = 0.0001
end

abstract type AbstractGrid end

struct CartesianGrid <: AbstractGrid
    ng::Int # Number of ghost cells
    dims::SVector{3, Int}
    x::Array{Float64, 3}; y::Array{Float64, 3}; z::Array{Float64, 3}
    volume::Array{Float64, 3}
    face_area_x::Array{Float64, 3}; face_area_y::Array{Float64, 3}; face_area_z::Array{Float64, 3}
    mask::Array{Bool, 3}
end

struct CurvilinearGrid <: AbstractGrid
    ng::Int # Number of ghost cells
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

mutable struct State
    tracers::Dict{Symbol, Array{Float64, 3}}
    _buffers::Dict{Symbol, Array{Float64, 3}}
    bed_mass::Dict{Symbol, Array{Float64, 2}}
    u::Array{Float64, 3}; v::Array{Float64, 3}; w::Array{Float64, 3}
    zeta::Array{Float64, 3}
    flux_x::Array{Float64, 3}
    flux_y::Array{Float64, 3}
    flux_z::Array{Float64, 3}
    temperature::Array{Float64, 3}; salinity::Array{Float64, 3}
    tss::Array{Float64, 3}; uvb::Array{Float64, 3}
    time::Float64
end

@kwdef struct PointSource
    i::Int; j::Int; k::Int
    tracer_name::Symbol
    influx_rate::Function
    relocate_if_dry::Bool = false
end

@kwdef struct FunctionalInteraction
    affected_tracers::Vector{Symbol}
    interaction_function::Function
end

abstract type BoundaryCondition end

@kwdef struct OpenBoundary <: BoundaryCondition
    side::Symbol
end

@kwdef struct RiverBoundary <: BoundaryCondition
    side::Symbol
    tracer_name::Symbol
    indices::UnitRange{Int}
    concentration::Function
    velocity::Function
end

@kwdef struct TidalBoundary <: BoundaryCondition
    side::Symbol
    inflow_concentrations::Function
end

struct HydrodynamicData
    filepath::String
    var_map::Dict{Symbol, String}
end

end # module ModelStructs