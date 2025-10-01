# src/StateModule.jl

module StateModule

export initialize_state

using ..HydrodynamicTransport.ModelStructs 
using NCDatasets

# This method for CartesianGrid is unchanged
function initialize_state(grid::CartesianGrid, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng
    nx, ny, nz = grid.dims
    
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx_tot, ny_tot, nz)
    end
    
    u = zeros(Float64, nx_tot + 1, ny_tot, nz)
    v = zeros(Float64, nx_tot, ny_tot + 1, nz)
    w = zeros(Float64, nx_tot, ny_tot, nz + 1)
    
    temperature = zeros(Float64, nx_tot, ny_tot, nz)
    salinity = zeros(Float64, nx_tot, ny_tot, nz)
    tss = zeros(Float64, nx_tot, ny_tot, nz)
    uvb = zeros(Float64, nx_tot, ny_tot, nz)

    return State(tracers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end

# --- FIX: This method now correctly calculates all staggered dimensions from first principles ---
function initialize_state(grid::CurvilinearGrid, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nz_w = length(grid.z_w)
    
    # Calculate total dimensions from physical dimensions and ghost cells
    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng

    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    end
    
    # Correctly allocate state arrays with staggered sizes
    # u lives on faces, so there is one more face than cells in the x-direction
    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    # v lives on faces, so there is one more face than cells in the y-direction
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz_w)
    
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)

    return State(tracers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end


# This method for reading from a NetCDF file is also updated for consistency
function initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng

    # Use physical dimensions from the grid struct, not the file, as the source of truth
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nz_w = length(grid.z_w)

    nx_rho_tot, ny_rho_tot = nx + 2*ng, ny + 2*ng
    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    end
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    
    u = zeros(Float64, nx_rho_tot + 1, ny_rho_tot, nz)
    v = zeros(Float64, nx_rho_tot, ny_rho_tot + 1, nz)
    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz_w)

    return State(tracers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end


end # module StateModule