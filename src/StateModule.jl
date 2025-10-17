# src/StateModule.jl

module StateModule

export initialize_state

using ..HydrodynamicTransport.ModelStructs 
using NCDatasets

function initialize_state(grid::CartesianGrid, tracer_names::NTuple{N, Symbol} where N; sediment_tracers::Vector{Symbol}=Symbol[])
    ng = grid.ng
    nx, ny, nz = grid.dims
    
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_tot, ny_tot, nz)
        tracers[name] = tracer_arr
        buffers[name] = zeros(size(tracer_arr))
    end
    
    # --- NEW: Initialize bed_mass dictionary ---
    bed_mass = Dict{Symbol, Array{Float64, 2}}()
    for name in sediment_tracers
        bed_mass[name] = zeros(Float64, nx_tot, ny_tot)
    end

    u = zeros(Float64, nx_tot + 1, ny_tot, nz)
    v = zeros(Float64, nx_tot, ny_tot + 1, nz)
    w = zeros(Float64, nx_tot, ny_tot, nz + 1)

    flux_x = zeros(size(u))
    flux_y = zeros(size(v))
    flux_z = zeros(size(w))
    
    temperature = zeros(Float64, nx_tot, ny_tot, nz)
    salinity = zeros(Float64, nx_tot, ny_tot, nz)
    tss = zeros(Float64, nx_tot, ny_tot, nz)
    uvb = zeros(Float64, nx_tot, ny_tot, nz)
    zeta = zeros(Float64, nx_tot, ny_tot, nz)

    # --- UPDATED: Pass bed_mass to the constructor ---
    return State(tracers, buffers, u, v, w, zeta, flux_x, flux_y, flux_z, temperature, salinity, tss, uvb, 0.0, bed_mass)
end

function initialize_state(grid::CurvilinearGrid, tracer_names::NTuple{N, Symbol} where N; sediment_tracers::Vector{Symbol}=Symbol[])
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
        tracers[name] = tracer_arr
        buffers[name] = zeros(size(tracer_arr))
    end
    
    # --- NEW: Initialize bed_mass dictionary ---
    bed_mass = Dict{Symbol, Array{Float64, 2}}()
    for name in sediment_tracers
        bed_mass[name] = zeros(Float64, nx_rho_tot, ny_rho_tot)
    end

    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz + 1)
    
    flux_x = zeros(size(u))
    flux_y = zeros(size(v))
    flux_z = zeros(size(w))
    
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    zeta = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)

    # --- UPDATED: Pass bed_mass to the constructor ---
    return State(tracers, buffers, u, v, w, zeta, flux_x, flux_y, flux_z, temperature, salinity, tss, uvb, 0.0, bed_mass)
end

function initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names::NTuple{N, Symbol} where N; sediment_tracers::Vector{Symbol}=Symbol[])
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng
    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
        tracers[name] = tracer_arr
        buffers[name] = zeros(size(tracer_arr))
    end
    
    # --- NEW: Initialize bed_mass dictionary ---
    bed_mass = Dict{Symbol, Array{Float64, 2}}()
    for name in sediment_tracers
        bed_mass[name] = zeros(Float64, nx_rho_tot, ny_rho_tot)
    end

    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz + 1)

    flux_x = zeros(size(u))
    flux_y = zeros(size(v))
    flux_z = zeros(size(w))
    
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    zeta = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)

    # --- UPDATED: Pass bed_mass to the constructor ---
    return State(tracers, buffers, u, v, w, zeta, flux_x, flux_y, flux_z, temperature, salinity, tss, uvb, 0.0, bed_mass)
end

end # module StateModule