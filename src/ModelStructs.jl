# src/ModelStructs.jl

module ModelStructs

export AbstractGrid, CartesianGrid, CurvilinearGrid, State, HydrodynamicData, PointSource, 
       BoundaryCondition, OpenBoundary, RiverBoundary, TidalBoundary, FunctionalInteraction,
       SedimentParams, DecayParams, OysterParams, OysterState, VirtualOyster

using StaticArrays
using Base: @kwdef

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
    _buffer1::Dict{Symbol, Array{Float64, 3}}  # Renamed from _buffers
    _buffer2::Dict{Symbol, Array{Float64, 3}}  # NEW: Second buffer for 3-sweep ADI
    u::Array{Float64, 3}; v::Array{Float64, 3}; w::Array{Float64, 3}
    zeta::Array{Float64, 3}
    flux_x::Array{Float64, 3} # Pre-allocated buffer for x-direction fluxes
    flux_y::Array{Float64, 3} # Pre-allocated buffer for y-direction fluxes
    flux_z::Array{Float64, 3} # Pre-allocated buffer for z-direction fluxes
    temperature::Array{Float64, 3}; salinity::Array{Float64, 3}
    tss::Array{Float64, 3}; uvb::Array{Float64, 3}
    time::Float64
    bed_mass::Dict{Symbol, Array{Float64, 2}} # Mass per unit area (kg/m^2)
end

@kwdef struct PointSource
    i::Int; j::Int; k::Int # Physical indices (1-based from the corner of the physical domain)
    tracer_name::Symbol
    influx_rate::Function # time -> value
    relocate_if_dry::Bool = false
end

@kwdef struct FunctionalInteraction
    affected_tracers::Vector{Symbol}
    interaction_function::Function
end

@kwdef struct DecayParams
    tracer_name::Symbol
    base_rate::Float64 = 0.0
    temp_ref::Float64 = 20.0
    temp_theta::Float64 = 1.0
    light_coeff::Float64 = 0.0 
end

@kwdef struct SedimentParams
    ws::Float64             # Settling velocity (m/s, positive downwards)
    erosion_rate::Float64   # A simple constant erosion rate (kg/m^2/s)
    tau_ce::Float64 = 0.05  # Critical shear stress for erosion (Pa)
end

@kwdef struct OysterParams
    wdw::Float64 = 0.5
    ϵ_free::Float64 = 0.01
    ϵ_sorbed::Float64 = 0.8
    tss_reject::Float64 = 5.0
    tss_clog::Float64 = 100.0
    kdep_20::Float64 = 0.23
    θ_dep::Float64 = 1.07
end

mutable struct OysterState
    c_oyster::Float64
end

struct VirtualOyster
    i::Int
    j::Int
    k::Int
    params::OysterParams
    state::OysterState
end

abstract type BoundaryCondition end

@kwdef struct OpenBoundary <: BoundaryCondition
    side::Symbol # :West, :East, :North, or :South
end

@kwdef struct RiverBoundary <: BoundaryCondition
    side::Symbol
    tracer_name::Symbol
    indices::UnitRange{Int} # Range of physical grid cells for the river (e.g., 40:50)
    concentration::Function # time -> value
    velocity::Function      # time -> value (normal to the boundary, positive is inflow)
end

@kwdef struct TidalBoundary <: BoundaryCondition
    side::Symbol
    # A function of time that returns a Dict of tracer concentrations for INFLOWING water
    inflow_concentrations::Function # e.g., t -> Dict(:Salinity => 35.0, :TracerX => 0.0)
end

struct HydrodynamicData
    filepath::String
    var_map::Dict{Symbol, String}
end

end # module ModelStructs
