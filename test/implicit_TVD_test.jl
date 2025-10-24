

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Test
using Revise
using HydrodynamicTransport
# We need to import the new function and the limiter
using ..HydrodynamicTransport.HorizontalTransportModule: advect_diffuse_tvd_implicit_x!
using ..HydrodynamicTransport.HorizontalTransportModule: advect_diffuse_tvd_implicit_y!
using ..HydrodynamicTransport.VerticalTransportModule: advect_diffuse_tvd_implicit_z!, get_dz_at_face, get_dz_centers
using ..HydrodynamicTransport.FluxLimitersModule: van_leer

# ==============================================================================
# --- TESTSET 6: TVD Implicit Advection Scheme ---
# ==============================================================================
@testset "6. TVD Implicit Advection" begin
    @info "Running Testset 6: TVD Implicit Advection..."
    
    # --- 1. Setup 1D Advection Test ---
    nx, ny, nz = 100, 1, 1
    Lx, Ly, Lz = 100.0, 1.0, 1.0
    grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
    state = initialize_state(grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]
    ng = grid.ng

    # Create an initial square pulse
    pulse_start, pulse_end = ng + 20, ng + 40
    C_in[pulse_start:pulse_end, ng+1, 1] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity
    u_vel = 1.0
    state.u .= u_vel
    
    # --- 2. Parameters ---
    Kh = 0.0 # Test ADVECTION ONLY
    dx = Lx / nx
    dt = 0.5 * dx / u_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        # Call the new function
        advect_diffuse_tvd_implicit_x!(C_next, C_current, state, grid, dt, Kh, van_leer)
        
        # Swap buffers for next iteration
        # --- FIX: Removed 'global' keyword ---
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current # This holds the result
    
    # --- 4. Verification ---
    
    @testset "Monotonicity (TVD Property)" begin
        # The key TVD property: no new maxima or minima are created.
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        # Check that the pulse has moved
        distance_moved = u_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        # Find the center of mass of the final pulse
        mass_weighted_index = sum(i * C_final[i+ng, ng+1, 1] for i in 1:nx)
        total_mass = sum(C_final[ng+1:end-ng, ng+1, 1])
        
        # Check if total_mass is near zero, which would mean the test failed
        @test total_mass > 1.0
        
        final_center = mass_weighted_index / total_mass
        
        # Check if the final center is close to the expected center
        @test isapprox(final_center, expected_center, rtol=0.1)
    end

    
end

@testset "6.5 TVD Implicit Advection (Curvilinear Grid)" begin
    @info "Running Testset 6.5: TVD Implicit Advection (Curvilinear Grid)..."
    
    # --- 1. Setup 1D Advection Test on a Curvilinear Grid ---
    nx, ny, nz = 100, 1, 1
    Lx, Ly, Lz = 100.0, 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    ng = 2 # Use the standard 2 ghost cells
    
    # --- Manually construct a simple, rectangular CurvilinearGrid ---
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    # Create grid metrics
    pm_full = fill(1.0 / dx, (nx_tot, ny_tot))
    pn_full = fill(1.0 / dy, (nx_tot, ny_tot))
    angle_full = zeros(nx_tot, ny_tot)
    h_full = fill(Lz, (nx_tot, ny_tot))
    mask_rho_full = ones(Bool, nx_tot, ny_tot)
    mask_u_full = ones(Bool, nx_tot + 1, ny_tot)
    mask_v_full = ones(Bool, nx_tot, ny_tot + 1)
    z_w_vec = [-Lz, 0.0]
    
    # Dummy coordinate arrays (not used by this test, but needed by struct)
    lon_rho, lat_rho = zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot)
    lon_u, lat_u = zeros(nx_tot + 1, ny_tot), zeros(nx_tot + 1, ny_tot)
    lon_v, lat_v = zeros(nx_tot, ny_tot + 1), zeros(nx_tot, ny_tot + 1)

    # Calculate face areas and volumes
    face_area_x = fill(dy * Lz, (nx_tot + 1, ny_tot, nz))
    face_area_y = fill(dx * Lz, (nx_tot, ny_tot + 1, nz))
    volume = fill(dx * dy * Lz, (nx_tot, ny_tot, nz))

    curv_grid = CurvilinearGrid(
        ng, nx, ny, nz,
        lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
        z_w_vec,
        pm_full, pn_full, angle_full, h_full,
        mask_rho_full, mask_u_full, mask_v_full,
        face_area_x, face_area_y, volume
    )
    
    state = initialize_state(curv_grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]

    # Create an initial square pulse
    pulse_start, pulse_end = ng + 20, ng + 40
    C_in[pulse_start:pulse_end, ng+1, 1] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity
    u_vel = 1.0
    state.u .= u_vel
    
    # --- 2. Parameters ---
    Kh = 0.0 # Test ADVECTION ONLY
    dt = 0.5 * dx / u_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        advect_diffuse_tvd_implicit_x!(C_next, C_current, state, curv_grid, dt, Kh, van_leer)
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current
    
    # --- 4. Verification (Identical to Cartesian test) ---
    
    @testset "Monotonicity (TVD Property)" begin
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        distance_moved = u_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        mass_weighted_index = sum(i * C_final[i+ng, ng+1, 1] for i in 1:nx)
        total_mass = sum(C_final[ng+1:end-ng, ng+1, 1])
        
        @test total_mass > 1.0
        final_center = mass_weighted_index / total_mass
        @test isapprox(final_center, expected_center, rtol=0.1)
    end
end

@testset "7. TVD Implicit Advection (Y-dir)" begin
    @info "Running Testset 7: TVD Implicit Advection (Y-dir)..."
    
    # --- 1. Setup 1D Advection Test (in Y-direction) ---
    nx, ny, nz = 1, 100, 1
    Lx, Ly, Lz = 1.0, 100.0, 1.0
    grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
    state = initialize_state(grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]
    ng = grid.ng

    # Create an initial square pulse in the Y-direction
    pulse_start, pulse_end = ng + 20, ng + 40
    C_in[ng+1, pulse_start:pulse_end, 1] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity in the Y-direction
    v_vel = 1.0
    state.v .= v_vel
    
    # --- 2. Parameters ---
    Kh = 0.0 # Test ADVECTION ONLY
    dy = Ly / ny
    dt = 0.5 * dy / v_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        advect_diffuse_tvd_implicit_y!(C_next, C_current, state, grid, dt, Kh, van_leer)
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current # This holds the result
    
    # --- 4. Verification ---
    
    @testset "Monotonicity (TVD Property)" begin
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        distance_moved = v_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        mass_weighted_index = sum(j * C_final[ng+1, j+ng, 1] for j in 1:ny)
        total_mass = sum(C_final[ng+1, ng+1:end-ng, 1])
        
        @test total_mass > 1.0
        final_center = mass_weighted_index / total_mass
        @test isapprox(final_center, expected_center, rtol=0.1)
    end
end

# ==============================================================================
# --- TESTSET 7.5: TVD Implicit Advection (Y-dir, Curvilinear) ---
# ==============================================================================
@testset "7.5 TVD Implicit Advection (Y-dir, Curvilinear)" begin
    @info "Running Testset 7.5: TVD Implicit Advection (Y-dir, Curvilinear)..."
    
    # --- 1. Setup 1D Advection Test on a Curvilinear Grid ---
    nx, ny, nz = 1, 100, 1
    Lx, Ly, Lz = 1.0, 100.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    ng = 2
    
    # --- Manually construct a simple, rectangular CurvilinearGrid ---
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    pm_full = fill(1.0 / dx, (nx_tot, ny_tot))
    pn_full = fill(1.0 / dy, (nx_tot, ny_tot))
    angle_full = zeros(nx_tot, ny_tot)
    h_full = fill(Lz, (nx_tot, ny_tot))
    mask_rho_full = ones(Bool, nx_tot, ny_tot)
    mask_u_full = ones(Bool, nx_tot + 1, ny_tot)
    mask_v_full = ones(Bool, nx_tot, ny_tot + 1)
    z_w_vec = [-Lz, 0.0]
    
    lon_rho, lat_rho = zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot)
    lon_u, lat_u = zeros(nx_tot + 1, ny_tot), zeros(nx_tot + 1, ny_tot)
    lon_v, lat_v = zeros(nx_tot, ny_tot + 1), zeros(nx_tot, ny_tot + 1)

    face_area_x = fill(dy * Lz, (nx_tot + 1, ny_tot, nz))
    face_area_y = fill(dx * Lz, (nx_tot, ny_tot + 1, nz))
    volume = fill(dx * dy * Lz, (nx_tot, ny_tot, nz))

    curv_grid = CurvilinearGrid(
        ng, nx, ny, nz,
        lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
        z_w_vec,
        pm_full, pn_full, angle_full, h_full,
        mask_rho_full, mask_u_full, mask_v_full,
        face_area_x, face_area_y, volume
    )
    
    state = initialize_state(curv_grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]

    # Create an initial square pulse in the Y-direction
    pulse_start, pulse_end = ng + 20, ng + 40
    C_in[ng+1, pulse_start:pulse_end, 1] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity in the Y-direction
    v_vel = 1.0
    state.v .= v_vel
    
    # --- 2. Parameters ---
    Kh = 0.0 # Test ADVECTION ONLY
    dt = 0.5 * dy / v_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        advect_diffuse_tvd_implicit_y!(C_next, C_current, state, curv_grid, dt, Kh, van_leer)
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current
    
    # --- 4. Verification (Identical to Cartesian test) ---
    
    @testset "Monotonicity (TVD Property)" begin
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        distance_moved = v_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        mass_weighted_index = sum(j * C_final[ng+1, j+ng, 1] for j in 1:ny)
        total_mass = sum(C_final[ng+1, ng+1:end-ng, 1])
        
        @test total_mass > 1.0
        final_center = mass_weighted_index / total_mass
        @test isapprox(final_center, expected_center, rtol=0.1)
    end
end

# ==============================================================================
# --- TESTSET 8: TVD Implicit Advection (Z-dir) ---
# ==============================================================================

@testset "8. TVD Implicit Advection (Z-dir)" begin
    @info "Running Testset 8: TVD Implicit Advection (Z-dir)..."
    
    # --- 1. Setup 1D Advection Test (in Z-direction) ---
    nx, ny, nz = 1, 1, 100
    Lx, Ly, Lz = 1.0, 1.0, 100.0
    grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
    state = initialize_state(grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]
    ng = grid.ng

    # Create an initial square pulse in the Z-direction
    # Note: Z-axis is often 1-indexed at the bottom
    pulse_start, pulse_end = 20, 40
    C_in[ng+1, ng+1, pulse_start:pulse_end] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity in the Z-direction
    w_vel = 1.0
    state.w .= w_vel # w-faces are at k=1...nz+1
    
    # --- 2. Parameters ---
    Kz = 0.0 # Test ADVECTION ONLY
    dz = Lz / nz
    dt = 0.5 * dz / w_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        advect_diffuse_tvd_implicit_z!(C_next, C_current, state, grid, dt, Kz, van_leer)
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current # This holds the result
    
    # --- 4. Verification ---
    
    @testset "Monotonicity (TVD Property)" begin
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        distance_moved = w_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        mass_weighted_index = sum(k * C_final[ng+1, ng+1, k] for k in 1:nz)
        total_mass = sum(C_final[ng+1, ng+1, 1:nz])
        
        @test total_mass > 1.0
        final_center = mass_weighted_index / total_mass
        @test isapprox(final_center, expected_center, rtol=0.1)
    end
end

# ==============================================================================
# --- TESTSET 8.5: TVD Implicit Advection (Z-dir, Curvilinear) ---
# ==============================================================================

@testset "8.5 TVD Implicit Advection (Z-dir, Curvilinear)" begin
    @info "Running Testset 8.5: TVD Implicit Advection (Z-dir, Curvilinear)..."
    
    # --- 1. Setup 1D Advection Test on a Curvilinear Grid ---
    nx, ny, nz = 1, 1, 100
    Lx, Ly, Lz = 1.0, 1.0, 100.0
    dx = Lx / nx
    dy = Ly / ny
    ng = 2
    
    # --- Manually construct a simple, rectangular CurvilinearGrid ---
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    pm_full = fill(1.0 / dx, (nx_tot, ny_tot))
    pn_full = fill(1.0 / dy, (nx_tot, ny_tot))
    angle_full = zeros(nx_tot, ny_tot)
    h_full = fill(Lz, (nx_tot, ny_tot))
    mask_rho_full = ones(Bool, nx_tot, ny_tot)
    mask_u_full = ones(Bool, nx_tot + 1, ny_tot)
    mask_v_full = ones(Bool, nx_tot, ny_tot + 1)
    
    # Create z_w vector, 101 faces from -100.0 to 0.0
    z_w_vec = collect(range(-Lz, 0.0, length=nz+1))
    dz = abs(z_w_vec[2] - z_w_vec[1]) # This is our dz_centers
    
    lon_rho, lat_rho = zeros(nx_tot, ny_tot), zeros(nx_tot, ny_tot)
    lon_u, lat_u = zeros(nx_tot + 1, ny_tot), zeros(nx_tot + 1, ny_tot)
    lon_v, lat_v = zeros(nx_tot, ny_tot + 1), zeros(nx_tot, ny_tot + 1)

    face_area_x = fill(dy * dz, (nx_tot + 1, ny_tot, nz))
    face_area_y = fill(dx * dz, (nx_tot, ny_tot + 1, nz))
    volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))

    curv_grid = CurvilinearGrid(
        ng, nx, ny, nz,
        lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
        z_w_vec,
        pm_full, pn_full, angle_full, h_full,
        mask_rho_full, mask_u_full, mask_v_full,
        face_area_x, face_area_y, volume
    )
    
    state = initialize_state(curv_grid, (:C,))
    
    C_in = state.tracers[:C]
    C_out = state._buffer1[:C]

    # Create an initial square pulse in the Z-direction
    pulse_start, pulse_end = 20, 40
    C_in[ng+1, ng+1, pulse_start:pulse_end] .= 1.0
    C_initial_max = maximum(C_in)
    C_initial_min = minimum(C_in)

    # Set a constant velocity in the Z-direction
    w_vel = 1.0
    state.w .= w_vel
    
    # --- 2. Parameters ---
    Kz = 0.0 # Test ADVECTION ONLY
    dt = 0.5 * dz / w_vel # CFL = 0.5
    n_steps = 20
    
    # --- 3. Run the Advection ---
    C_current = C_in
    C_next = C_out
    
    for _ in 1:n_steps
        advect_diffuse_tvd_implicit_z!(C_next, C_current, state, curv_grid, dt, Kz, van_leer)
        C_current, C_next = C_next, C_current 
    end
    
    C_final = C_current
    
    # --- 4. Verification (Identical to Cartesian test) ---
    
    @testset "Monotonicity (TVD Property)" begin
        @test maximum(C_final) <= C_initial_max
        @test minimum(C_final) >= C_initial_min
    end
    
    @testset "Advection" begin
        distance_moved = w_vel * dt * n_steps
        expected_center = (20 + 40) / 2.0 + distance_moved
        
        mass_weighted_index = sum(k * C_final[ng+1, ng+1, k] for k in 1:nz)
        total_mass = sum(C_final[ng+1, ng+1, 1:nz])
        
        @test total_mass > 1.0
        final_center = mass_weighted_index / total_mass
        @test isapprox(final_center, expected_center, rtol=0.1)
    end
end