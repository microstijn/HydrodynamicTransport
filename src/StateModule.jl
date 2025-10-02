# src/StateModule.jl

module StateModule

export initialize_state

using ..HydrodynamicTransport.ModelStructs 
using NCDatasets

# This method for CartesianGrid is now updated
function initialize_state(grid::CartesianGrid, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng
    nx, ny, nz = grid.dims
    
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}() # Create the buffers dictionary
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_tot, ny_tot, nz)
        tracers[name] = tracer_arr
        # --- FIX: Replace similar() with zeros() for predictable initialization ---
        buffers[name] = zeros(size(tracer_arr))
    end
    
    u = zeros(Float64, nx_tot + 1, ny_tot, nz)
    v = zeros(Float64, nx_tot, ny_tot + 1, nz)
    w = zeros(Float64, nx_tot, ny_tot, nz + 1)
    
    temperature = zeros(Float64, nx_tot, ny_tot, nz)
    salinity = zeros(Float64, nx_tot, ny_tot, nz)
    tss = zeros(Float64, nx_tot, ny_tot, nz)
    uvb = zeros(Float64, nx_tot, ny_tot, nz)

    return State(tracers, buffers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end

# This method for CurvilinearGrid is now updated
function initialize_state(grid::CurvilinearGrid, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nz_w = length(grid.z_w)
    
    # Calculate total dimensions from physical dimensions and ghost cells
    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}() # Create the buffers dictionary
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
        tracers[name] = tracer_arr
        # --- FIX: Replace similar() with zeros() for predictable initialization ---
        buffers[name] = zeros(size(tracer_arr))
    end
    
    # Correctly allocate state arrays with staggered sizes
    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz_w)
    
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)

    return State(tracers, buffers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end


# This method for reading from a NetCDF file is now updated
function initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng

    # Use physical dimensions from the grid struct, not the file, as the source of truth
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nz_w = length(grid.z_w)

    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng
    tracers = Dict{Symbol, Array{Float64, 3}}()
    buffers = Dict{Symbol, Array{Float64, 3}}() # Create the buffers dictionary
    for name in tracer_names
        tracer_arr = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
        tracers[name] = tracer_arr
        # --- FIX: Replace similar() with zeros() for predictable initialization ---
        buffers[name] = zeros(size(tracer_arr))
    end
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    
    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz_w)

    return State(tracers, buffers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end


end # module StateModule