# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using Test

# Explicitly import the functions we are testing
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.StateModule: initialize_state
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!

@testset "Cell-Face Blocking Logic" begin
    # --- 2. Create a simple, uniform Curvilinear Grid ---
    NG = 2
    nx, ny, nz = 5, 5, 1
    Lx, Ly = 100.0, 100.0
    dx, dy, dz = Lx / nx, Ly / ny, 1.0

    # Total dimensions including ghost cells
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

    # Create simple, uniform grid metrics
    pm = ones(Float64, nx_tot, ny_tot) .* (1/dx)
    pn = ones(Float64, nx_tot, ny_tot) .* (1/dy)
    angle = zeros(Float64, nx_tot, ny_tot)
    
    # Bathymetry: Deep everywhere except for one shallow cell
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    shallow_cell_i_phys, shallow_cell_j_phys = 2, 3
    shallow_cell_i_glob, shallow_cell_j_glob = shallow_cell_i_phys + NG, shallow_cell_j_phys + NG
    h[shallow_cell_i_glob, shallow_cell_j_glob] = 0.5

    # Dummy arrays for unused grid fields in this test
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

    # --- 3. Initialize State ---
    state = initialize_state(grid, (:C,))
    
    # Set a constant velocity field from left to right
    state.u .= 1.0
    
    # Sea surface height is zero everywhere
    state.zeta .= 0.0
    
    # Place a tracer in the shallow cell
    C = state.tracers[:C]
    C[shallow_cell_i_glob, shallow_cell_j_glob, 1] = 100.0
    
    # The cell to the right of the shallow cell should start with zero tracer
    downstream_cell_i_glob = shallow_cell_i_glob + 1
    downstream_cell_j_glob = shallow_cell_j_glob
    @test C[downstream_cell_i_glob, downstream_cell_j_glob, 1] == 0.0

    # --- 4. Run Transport with Blocking Enabled ---
    dt = 1.0
    
    # The total depth of the shallow cell is h + zeta = 0.5 + 0.0 = 0.5
    # Set D_crit higher than this depth to trigger blocking
    D_crit = 1.0
    
    # We only need to test one transport scheme, UP3 is fine
    horizontal_transport!(state, grid, dt, :TVD, D_crit)
    
    # --- 5. Assertions ---
    # The tracer should NOT have been transported to the downstream cell
    # because the flux out of the shallow cell should have been blocked.
    @test C[downstream_cell_i_glob, downstream_cell_j_glob, 1] == 0.0
    
    println("Cell-face blocking test passed: Flux was correctly blocked from the shallow cell.")

    # --- 6. Now, run transport WITHOUT blocking ---
    # Reset the tracer concentration
    C .= 0.0
    C[shallow_cell_i_glob, shallow_cell_j_glob, 1] = 100.0
    
    # Set D_crit to 0.0, which should disable the blocking for any cell with depth > 0
    D_crit_disabled = 0.0
    horizontal_transport!(state, grid, dt, :TVD, D_crit_disabled)
    
    # --- 7. Assertions for Non-Blocking Case ---
    # The tracer SHOULD now have been transported to the downstream cell
    @test C[downstream_cell_i_glob, downstream_cell_j_glob, 1] > 0.0
    
    println("Cell-face blocking test passed: Flux was correctly allowed when D_crit was disabled.")
end