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
    boundary_conditions = Vector{BoundaryCondition}()
    horizontal_transport!(state, grid, dt, :TVD, D_crit, boundary_conditions)
    
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
    horizontal_transport!(state, grid, dt, :TVD, D_crit_disabled, boundary_conditions)
    
    # --- 7. Assertions for Non-Blocking Case ---
    # The tracer SHOULD now have been transported to the downstream cell
    @test C[downstream_cell_i_glob, downstream_cell_j_glob, 1] > 0.0
    
    println("Cell-face blocking test passed: Flux was correctly allowed when D_crit was disabled.")
end

@testset "Implicit ADI Mass Conservation" begin
    # --- 1. Setup Grid and State (similar to above) ---
    NG = 2
    nx, ny, nz = 10, 10, 1
    Lx, Ly = 200.0, 200.0
    dx, dy, dz = Lx / nx, Ly / ny, 1.0
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG

    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* 10.0
    volume = ones(Float64, nx_tot, ny_tot, nz) .* (dx * dy * dz)
    
    # Dummy arrays for unused fields
    zeros_arr = zeros(nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    trues_arr_u = trues(nx_tot + 1, ny_tot)
    trues_arr_v = trues(nx_tot, ny_tot + 1)
    face_area_x = ones(Float64, nx_tot + 1, ny_tot, nz) .* (dy * dz)
    face_area_y = ones(Float64, nx_tot, ny_tot + 1, nz) .* (dx * dz)
    z_w = [-dz, 0.0]

    grid = CurvilinearGrid(NG, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                           z_w, pm, pn, zeros_arr, h,
                           trues_arr_rho, trues_arr_u, trues_arr_v,
                           face_area_x, face_area_y, volume)

    state = initialize_state(grid, (:C,))
    state.u .= 0.5  # Constant velocity U
    state.v .= 0.5  # Constant velocity V
    state.zeta .= 0.0

    # --- 2. Set Initial Condition and Calculate Initial Mass ---
    C = state.tracers[:C]
    C_phys = view(C, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
    C_phys[4:6, 4:6, 1] .= 100.0 # Initial patch of tracer

    volume_phys = view(grid.volume, (NG+1):(nx+NG), (NG+1):(ny+NG), 1:nz)
    initial_mass = sum(C_phys .* volume_phys)
    @test initial_mass > 0.0

    # --- 3. Run Transport with Implicit ADI Scheme ---
    # Use a timestep that would be unstable for an explicit scheme (Courant > 1)
    # Courant number = u*dt/dx + v*dt/dy = 0.5*dt/20 + 0.5*dt/20 = 0.05*dt
    # To make Courant > 1, need dt > 20. Let's use dt = 30.
    dt = 30.0
    D_crit = 0.0
    boundary_conditions = Vector{BoundaryCondition}()

    # Run for a few steps to allow for transport
    for _ in 1:5
        horizontal_transport!(state, grid, dt, :ImplicitADI, D_crit, boundary_conditions)
    end

    # --- 4. Calculate Final Mass and Assert Conservation ---
    final_mass = sum(C_phys .* volume_phys)
    
    @test initial_mass ≈ final_mass rtol=1e-12
    
    println("Implicit ADI test passed: Simulation ran and mass was conserved.")
end