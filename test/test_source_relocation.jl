# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using Test

# Explicitly import the functions and structs we are testing
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.StateModule: initialize_state
using HydrodynamicTransport.SourceSinkModule: source_sink_terms!

@testset "Source Relocation Logic" begin
    # --- 2. Create a simple Curvilinear Grid for testing ---
    NG = 2
    nx, ny, nz = 5, 5, 1
    Lx, Ly = 100.0, 100.0
    dx, dy, dz = Lx / nx, Ly / ny, 10.0 # Use a realistic depth for volume calculation

    # Total dimensions including ghost cells
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

    # Create simple, uniform grid metrics
    pm = ones(Float64, nx_tot, ny_tot) .* (1/dx)
    pn = ones(Float64, nx_tot, ny_tot) .* (1/dy)
    angle = zeros(Float64, nx_tot, ny_tot)
    
    # Bathymetry: A dry cell surrounded by wet cells
    h = ones(Float64, nx_tot, ny_tot) .* 10.0 # Wet everywhere by default
    dry_cell_i_phys, dry_cell_j_phys = 3, 3
    dry_cell_i_glob, dry_cell_j_glob = dry_cell_i_phys + NG, dry_cell_j_phys + NG
    h[dry_cell_i_glob, dry_cell_j_glob] = 0.1 # This cell is very shallow ("dry")

    # Dummy arrays for unused grid fields
    lon_rho = zeros(nx_tot, ny_tot)
    lat_rho = zeros(nx_tot, ny_tot)
    mask_rho = trues(nx_tot, ny_tot)
    mask_u = trues(nx_tot + 1, ny_tot)
    mask_v = trues(nx_tot, ny_tot + 1)
    z_w = [-dz, 0.0]

    # Calculate volume and face areas
    face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
    face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)
    volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)

    grid = CurvilinearGrid(NG, nx, ny, nz, lon_rho, lat_rho, lon_rho, lat_rho, lon_rho, lat_rho, 
                           z_w, pm, pn, angle, h,
                           mask_rho, mask_u, mask_v,
                           face_area_x, face_area_y, volume)

    # --- 3. Test Case 1: Relocation Enabled ---
    @testset "Relocation Enabled" begin
        # --- Initialize State ---
        state = initialize_state(grid, (:TestTracer,))
        state.zeta .= 0.0 # Sea surface height is zero
        
        # --- Define Point Source ---
        source = PointSource(
            i = dry_cell_i_phys,
            j = dry_cell_j_phys,
            k = 1,
            tracer_name = :TestTracer,
            influx_rate = t -> 100.0, # Simple constant rate
            relocate_if_dry = true
        )

        # --- Run the source term function ---
        dt = 1.0
        # D_crit is higher than the dry cell's depth (0.1), so it should be considered dry
        D_crit = 0.5 
        source_sink_terms!(state, grid, [source], 0.0, dt, D_crit)

        # --- Assertions ---
        C = state.tracers[:TestTracer]
        
        # The original "dry" cell should have received no tracer
        @test C[dry_cell_i_glob, dry_cell_j_glob, 1] == 0.0
        
        # The search algorithm should find the first wet neighbor.
        # Based on the loop in `_find_nearest_wet_neighbor`, this will be at (-1, -1) offset.
        neighbor_i_phys, neighbor_j_phys = dry_cell_i_phys - 1, dry_cell_j_phys - 1
        neighbor_i_glob, neighbor_j_glob = neighbor_i_phys + NG, neighbor_j_phys + NG
        
        # The tracer should be in the identified wet neighbor cell
        @test C[neighbor_i_glob, neighbor_j_glob, 1] > 0.0
        
        # Verify the concentration is correct
        expected_concentration = (100.0 * dt) / volume[neighbor_i_glob, neighbor_j_glob, 1]
        @test C[neighbor_i_glob, neighbor_j_glob, 1] ≈ expected_concentration

        println("✅ Relocation Enabled Test Passed: Source was correctly moved to the nearest wet neighbor.")
    end

    # --- 4. Test Case 2: Relocation Disabled ---
    @testset "Relocation Disabled" begin
        # --- Initialize State ---
        state = initialize_state(grid, (:TestTracer,))
        state.zeta .= 0.0
        
        # --- Define Point Source (relocate_if_dry is false by default) ---
        source = PointSource(
            i = dry_cell_i_phys,
            j = dry_cell_j_phys,
            k = 1,
            tracer_name = :TestTracer,
            influx_rate = t -> 100.0
        )

        # --- Run the source term function ---
        dt = 1.0
        D_crit = 0.5 
        source_sink_terms!(state, grid, [source], 0.0, dt, D_crit)

        # --- Assertions ---
        C = state.tracers[:TestTracer]

        # The tracer should be added to the original "dry" cell because relocation is off
        @test C[dry_cell_i_glob, dry_cell_j_glob, 1] > 0.0

        # Verify the concentration is correct
        expected_concentration = (100.0 * dt) / volume[dry_cell_i_glob, dry_cell_j_glob, 1]
        @test C[dry_cell_i_glob, dry_cell_j_glob, 1] ≈ expected_concentration

        # Check that a neighboring cell did NOT receive any tracer
        neighbor_i_glob, neighbor_j_glob = (dry_cell_i_phys - 1) + NG, (dry_cell_j_phys - 1) + NG
        @test C[neighbor_i_glob, neighbor_j_glob, 1] == 0.0

        println("✅ Relocation Disabled Test Passed: Source was correctly placed in the original dry cell.")
    end
end