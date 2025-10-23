# src/SettlingModule.jl

module SettlingModule

export apply_settling!

using ..HydrodynamicTransport.ModelStructs
using LinearAlgebra

"""
    apply_settling!(state::State, grid::AbstractGrid, dt::Float64, sediment_tracers::Dict{Symbol, SedimentParams})

Applies sediment settling using a numerically stable implicit scheme.

This function solves the 1D vertical advection equation for settling.
The implicit, first-order upwind formulation is used:
(1 + ν_k) * C_k^{n+1} - ν_k * C_{k+1}^{n+1} = C_k^n
where ν is the Courant number (ws * dt / dz).

This creates a bidiagonal system that is solved for each water column.
The flux of sediment out of the bottom-most water cell is calculated and
returned as the deposition flux to the bed.

# Arguments
- `state::State`: The model state, modified in-place.
- `grid::AbstractGrid`: The computational grid.
- `dt::Float64`: The time step duration.
- `sediment_tracers`: A Dict mapping tracer symbols to their parameters.

# Returns
- `deposition_fluxes::Dict{Symbol, Array{Float64, 2}}`: A dictionary mapping
  each sediment tracer to a 2D array of deposition fluxes (in mass/m^2/s)
  at the seabed for each (i,j) location.
"""
function apply_settling!(state::State, grid::AbstractGrid, dt::Float64, sediment_tracers::Dict{Symbol, SedimentParams})
    # FIX: Ensure the correct type is returned even if the dictionary is empty.
    if isempty(sediment_tracers); return Dict{Symbol, Array{Float64, 2}}(); end

    ng = grid.ng
    nx, ny, nz = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)
    if nz <= 1; return Dict{Symbol, Array{Float64, 2}}(); end

    deposition_fluxes = Dict{Symbol, Array{Float64, 2}}()

    for (tracer_name, params) in sediment_tracers
        if !haskey(state.tracers, tracer_name); continue; end

        C = state.tracers[tracer_name]
        ws = params.ws
        deposition_fluxes[tracer_name] = zeros(Float64, nx + 2*ng, ny + 2*ng)

        # Pre-allocate vectors for the bidiagonal system
        b = Vector{Float64}(undef, nz)     # main diagonal
        c = Vector{Float64}(undef, nz - 1) # super-diagonal (influence from cell k+1 on k)
        d = Vector{Float64}(undef, nz)     # RHS (current concentration C_n)

        @inbounds for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng

            # --- 1. Construct the bidiagonal system for the column ---
            for k in 1:nz
                dz_k = isa(grid, CartesianGrid) ? grid.volume[i_glob, j_glob, k] / grid.face_area_z[i_glob, j_glob, k] : abs(grid.z_w[k+1] - grid.z_w[k])
                courant_k = ws * dt / dz_k

                b[k] = 1.0 + courant_k
                if k < nz
                    # Influence of cell k+1 (above) on cell k (below)
                    c[k] = -courant_k
                end
                d[k] = C[i_glob, j_glob, k]
            end
            
            # --- 2. Solve the system ---
            A = Bidiagonal(b, c, :U) # Upper bidiagonal system
            solution = A \ d
            
            # --- 3. Store the result back into the tracer array ---
            view(C, i_glob, j_glob, :) .= solution

            # --- 4. Calculate deposition flux from the new bottom cell concentration ---
            new_C_bottom = solution[1]
            deposition_fluxes[tracer_name][i_glob, j_glob] = ws * new_C_bottom
        end
    end

    return deposition_fluxes
end

end # module SettlingModule
