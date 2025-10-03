using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!

println("--- Curvilinear Boundary Debugging Script ---")

# --- 2. Configuration ---
const NG = 2
const NC_URL = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
const HYDRO_DATA = HydrodynamicData(NC_URL, Dict(:u => "u", :v => "v", :time => "ocean_time"))

# Wrap in a try-catch block to handle potential network errors
try
    # --- 3. Initialize Grid, State, and Hydrodynamics ---
    println("Initializing grid and state from NetCDF file...")
    grid = initialize_curvilinear_grid(NC_URL; ng=NG)
    ds = NCDataset(NC_URL)
    state = initialize_state(grid, ds, (:Salinity,))

    println("Loading velocity field for the first time step...")
    update_hydrodynamics!(state, grid, ds, HYDRO_DATA, 0.0)
    
    # Set a uniform "brackish" salinity in the physical domain to start
    C_salinity = state.tracers[:Salinity]
    nx_phys, ny_phys = grid.nx, grid.ny
    C_phys_view = view(C_salinity, NG+1:nx_phys+NG, NG+1:ny_phys+NG, :)
    C_phys_view .= 10.0
    
    # --- 4. Define Boundary Conditions ---
    # A river of freshwater (salinity=0) on the West boundary
    river_indices = 250:300
    river_bc = RiverBoundary(side=:West, tracer_name=:Salinity, indices=river_indices, concentration=t->0.0, velocity=t->0.5)

    # A tidal boundary for the open sea on the East
    tidal_bc = TidalBoundary(side=:East, inflow_concentrations=t->Dict(:Salinity => 35.0))
    
    all_bcs = [river_bc, tidal_bc]

    println("\n--- State Before Applying BCs ---")
    println("Physical domain is 10.0. Ghost cells are 0.0.")
    
    # --- 5. Apply Boundary Conditions ---
    apply_boundary_conditions!(state, grid, all_bcs)
    
    println("\n--- 6. Verification Report ---")

    # --- Verify River Boundary on the West ---
    println("\n--- Verifying River Boundary on the West (Physical Y-indices $(river_indices)) ---")
    river_ok = true
    for j_phys in river_indices
        j_glob = j_phys + NG
        ghost_cells_are_zero = all(C_salinity[1:NG, j_glob, 1] .== 0.0)
        velocity_is_set = state.u[NG+1, j_glob, 1] == 0.5
        if !(ghost_cells_are_zero && velocity_is_set)
            river_ok = false
            println("❌ FAILED at j_phys=$j_phys")
        end
    end
    if river_ok
        println("✅ River boundary appears to be set correctly.")
    end

    # --- Verify Tidal Boundary on the East ---
    println("\n--- Verifying Tidal Boundary on the East (Point-by-Point) ---")
    tidal_ok = true
    for j_phys in 1:ny_phys
        j_glob = j_phys + NG
        
        velocity_at_face = state.u[nx_phys+NG+1, j_glob, 1]
        phys_cell_val = C_salinity[nx_phys+NG, j_glob, 1]
        ghost_cell_val = C_salinity[nx_phys+NG+1, j_glob, 1]
        
        flow_dir = velocity_at_face < 0 ? "INFLOW" : "OUTFLOW"
        
        correct = false
        expected_val = 0.0
        if flow_dir == "INFLOW"
            expected_val = 35.0
            correct = isapprox(ghost_cell_val, expected_val)
        else # OUTFLOW
            expected_val = phys_cell_val
            correct = isapprox(ghost_cell_val, expected_val)
        end
        
        if !correct
            tidal_ok = false
            println("❌ FAILED at j_phys=$j_phys ($flow_dir): Vel=$(round(velocity_at_face, digits=2)). Phys Cell=$phys_cell_val. Expected Ghost=$expected_val, Got Ghost=$ghost_cell_val")
        end
    end
    
    if tidal_ok
        println("✅ Tidal boundary logic appears to be correct for all points.")
    end

    close(ds)

catch e
     if isa(e, NCDatasets.NetCDFError)
        @warn "Skipping script: Could not access remote NetCDF data."
    else
        rethrow(e)
    end
end

println("\n--- Debugging Script Finished ---")