# src/StateModule.jl

module StateModule

export initialize_state

using ..HydrodynamicTransport.ModelStructs
using NCDatasets

# This method for CartesianGrid is updated for consistency with the new architecture.
function initialize_state(grid::CartesianGrid, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng
    nx, ny, nz = grid.dims
    
    # Total dimensions including ghost cells
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

    # Initialize time to 0.0
    return State(tracers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end

# --- Refactored State Initializer for Curvilinear Grids ---
function initialize_state(grid::CurvilinearGrid, ds::NCDataset, tracer_names::NTuple{N, Symbol} where N)
    ng = grid.ng

    # --- 1. Read Physical Dimensions from NetCDF file ---
    nx_rho, ny_rho = ds.dim["xi_rho"], ds.dim["eta_rho"]
    nx_u,   ny_u   = ds.dim["xi_u"],   ds.dim["eta_u"]
    nx_v,   ny_v   = ds.dim["xi_v"],   ds.dim["eta_v"]
    nz = ds.dim["s_rho"]
    nz_w = ds.dim["s_w"]

    # --- 2. Allocate State Arrays with Ghost Cells ---

    # Rho-point arrays (tracers, salinity, etc.)
    nx_rho_tot, ny_rho_tot = nx_rho + 2*ng, ny_rho + 2*ng
    tracers = Dict{Symbol, Array{Float64, 3}}()
    for name in tracer_names
        tracers[name] = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    end
    temperature = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    salinity = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    tss = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)
    uvb = zeros(Float64, nx_rho_tot, ny_rho_tot, nz)

    # Staggered arrays (u, v, w)
    # The array sizes must match the full grid arrays from GridModule, including ghost cells.
    # Note: For a standard Arakawa-C grid, u would have ny_rho_tot and v would have nx_rho_tot.
    # We follow the dimension names from the file for consistency with GridModule.
    nx_u_tot, ny_u_tot = nx_u + 2*ng, ny_u + 2*ng
    u = zeros(Float64, nx_u_tot, ny_u_tot, nz)

    nx_v_tot, ny_v_tot = nx_v + 2*ng, ny_v + 2*ng
    v = zeros(Float64, nx_v_tot, ny_v_tot, nz)

    w = zeros(Float64, nx_rho_tot, ny_rho_tot, nz_w)

    # --- 3. Construct and Return the State Struct ---
    return State(tracers, u, v, w, temperature, salinity, tss, uvb, 0.0)
end


end # module StateModule

