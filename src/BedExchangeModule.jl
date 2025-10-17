# src/BedExchangeModule.jl

module BedExchangeModule

export bed_exchange!

using ..HydrodynamicTransport.ModelStructs

"""
    bed_exchange!(state, grid, dt, deposition_fluxes, sediment_tracers)

Handles the exchange of sediment between the bottom water layer and the seabed.

This function performs two main actions for each sediment tracer:
1.  **Deposition**: It takes the depositional flux from the settling module,
    calculates the total deposited mass, and adds it to the `state.bed_mass`.
2.  **Erosion**: It calculates an erosion flux (resuspension) based on parameters,
    removes that mass from the `state.bed_mass`, and adds it back into the
    bottom water cell's concentration.

# Arguments
- `state::State`: The model state, which is modified in-place.
- `grid::AbstractGrid`: The computational grid.
- `dt::Float64`: The time step duration.
- `deposition_fluxes::Dict{Symbol, Array{Float64, 2}}`: The output from the `apply_settling!` function.
- `sediment_tracers::Dict{Symbol, SedimentParams}`: A dictionary mapping sediment
  tracer names to their parameters.

# Returns
- `nothing`: The function modifies `state.tracers` and `state.bed_mass` in-place.
"""
function bed_exchange!(state::State, grid::AbstractGrid, dt::Float64, deposition_fluxes::Dict{Symbol, Array{Float64, 2}}, sediment_tracers::Dict{Symbol, SedimentParams})
    ng = grid.ng
    nx, ny, _ = isa(grid, CartesianGrid) ? grid.dims : (grid.nx, grid.ny, grid.nz)

    for (tracer_name, params) in sediment_tracers
        if !haskey(state.tracers, tracer_name); continue; end

        C = state.tracers[tracer_name]
        bed = state.bed_mass[tracer_name]
        deposition = deposition_fluxes[tracer_name]

        @inbounds for j_phys in 1:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            
            # --- 1. Deposition ---
            deposited_mass_per_area = deposition[i_glob, j_glob] * dt
            bed[i_glob, j_glob] += deposited_mass_per_area

            # --- 2. Erosion (Simplified) ---
            # A more realistic model would calculate bottom shear stress here.
            erosion_flux = params.erosion_rate # in kg/m^2/s
            
            # Ensure we don't erode more mass than is available on the bed
            max_erodible_mass_per_area = bed[i_glob, j_glob]
            actual_eroded_mass_per_area = min(erosion_flux * dt, max_erodible_mass_per_area)
            
            # Update the bed
            bed[i_glob, j_glob] -= actual_eroded_mass_per_area
            
            # --- 3. Add eroded mass to the bottom water cell ---
            bottom_cell_volume = grid.volume[i_glob, j_glob, 1]
            if bottom_cell_volume > 1e-9
                # Horizontal area of the cell from grid metrics pm and pn
                cell_area = 1.0 / (grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob])
                
                # Change in concentration = (Mass per Area * Area) / Volume
                dC = (actual_eroded_mass_per_area * cell_area) / bottom_cell_volume
                C[i_glob, j_glob, 1] += dC
            end
        end
    end
end

end # module BedExchangeModule