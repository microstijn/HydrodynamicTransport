# src/ModelStructs.jl

module ModelStructs

export FT, AbstractGrid, CartesianGrid, CurvilinearGrid, State, HydrodynamicData, HydroSlabCache, PointSource,
       BoundaryCondition, OpenBoundary, RiverBoundary, TidalBoundary, FunctionalInteraction,
       SedimentParams, DecayParams, OysterParams, OysterState, VirtualOyster

using StaticArrays
using Base: @kwdef

abstract type AbstractGrid end

# Storage precision for the tracer / flux / bed-mass fields. Transport is memory-bandwidth-bound,
# so storing these (the O(n_tracers) bulk that is streamed every step) in Float32 roughly halves
# the dominant memory traffic for a ~1.5-2x speed-up. The arithmetic is still done in Float64
# inside the kernels (scratch + intermediates promote); only storage is reduced. Hydro/environment
# fields (u, v, w, zeta, T, S, ...) stay Float64 — they are read from disk and drive the CFL.
const FT = Float32

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
    tracers::Dict{Symbol, Array{FT, 3}}
    _buffer1::Dict{Symbol, Array{FT, 3}}  # Renamed from _buffers
    _buffer2::Dict{Symbol, Array{FT, 3}}  # NEW: Second buffer for 3-sweep ADI
    u::Array{Float64, 3}; v::Array{Float64, 3}; w::Array{Float64, 3}
    zeta::Array{Float64, 3}
    flux_x::Array{FT, 3} # Pre-allocated buffer for x-direction fluxes
    flux_y::Array{FT, 3} # Pre-allocated buffer for y-direction fluxes
    flux_z::Array{FT, 3} # Pre-allocated buffer for z-direction fluxes
    # Per-task scratch flux buffers for tracer-parallel horizontal transport. Lazily filled
    # (one set per parallel chunk) on first use; reused across steps. Scratch only -> not copied.
    flux_x_pool::Vector{Array{FT, 3}}
    flux_y_pool::Vector{Array{FT, 3}}
    temperature::Array{Float64, 3}; salinity::Array{Float64, 3}
    tss::Array{Float64, 3}; uvb::Array{Float64, 3}
    time::Float64
    bed_mass::Dict{Symbol, Array{FT, 2}} # Mass per unit area (kg/m^2)
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

# In-memory cache of the bracketing hydro time-slabs, so update_hydrodynamics! does not
# re-read the NetCDF on every timestep (only when the bracketing time index changes).
mutable struct HydroSlabCache
    time_seconds::Union{Nothing, Vector{Float64}}            # converted time axis, computed once
    slabs::Dict{Int, Dict{Symbol, Array{Float64}}}          # time-index -> (field -> coalesced slab)
end
HydroSlabCache() = HydroSlabCache(nothing, Dict{Int, Dict{Symbol, Array{Float64}}}())

struct HydrodynamicData
    filepath::String
    var_map::Dict{Symbol, String}
    cache::HydroSlabCache
end
# Backward-compatible constructor (all existing 2-arg call sites get a fresh cache).
HydrodynamicData(filepath::String, var_map::Dict{Symbol, String}) =
    HydrodynamicData(filepath, var_map, HydroSlabCache())

end # module ModelStructs
