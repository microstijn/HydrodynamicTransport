# test/sediment_tests.jl
using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))
using Revise
using HydrodynamicTransport
using HydrodynamicTransport.SettlingModule
using HydrodynamicTransport.BedExchangeModule
using HydrodynamicTransport.HorizontalTransportModule
using Test
# test/sediment_tests.jl

# test/sediment_tests.jl

@testset "6. Sediment Transport Modules (Curvilinear Grid)" begin
    @info "Running Testset 6: Sediment Settling and Bed Exchange on a Curvilinear Grid..."

    # --- 1. Grid and State Setup ---
    ng = 2
    nx, ny, nz = 1, 1, 10
    Lx, Ly, Lz = 10.0, 10.0, 10.0 # Total dimensions
    dx, dy = Lx / nx, Ly / ny
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* Lz
    
    zeros_arr = zeros(Float64, nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    
    trues_arr_u = trues(nx + 1 + 2*ng, ny + 2*ng)
    trues_arr_v = trues(nx + 2*ng, ny + 1 + 2*ng)

    z_w = collect(range(-Lz, 0, length=nz+1))
    
    volume = zeros(Float64, nx_tot, ny_tot, nz)
    face_area_x = zeros(Float64, nx_tot + 1, ny_tot, nz)
    face_area_y = zeros(Float64, nx_tot, ny_tot + 1, nz)

    for k in 1:nz
        dz = abs(z_w[k+1] - z_w[k]) # Should be 1.0m for this setup
        volume[:, :, k] .= dx * dy * dz
        face_area_x[:, :, k] .= dy * dz
        face_area_y[:, :, k] .= dx * dz
    end

    grid = CurvilinearGrid(ng, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                           z_w, pm, pn, zeros_arr, h,
                           trues_arr_rho, trues_arr_u, trues_arr_v,
                           face_area_x, face_area_y, volume)

    # --- 2. Parameters and Helper Functions ---
    tracer_name = :FineSand
    sediment_params = Dict(
        tracer_name => SedimentParams(
            ws = 0.01,           # 1 cm/s settling velocity
            erosion_rate = 1e-6  # 1 mg/m^2/s erosion rate
        )
    )

    function get_total_water_mass(state, grid, tracer_name)
        return sum(state.tracers[tracer_name] .* grid.volume)
    end
    
    function get_total_bed_mass(state, grid, tracer_name)
        cell_area = dx * dy
        # Sum over all physical cells (in this case, just one)
        return sum(view(state.bed_mass[tracer_name], ng+1:nx+ng, ng+1:ny+ng)) * cell_area
    end

    # ==============================================================================
    # --- 6.1: Tests for SettlingModule.jl ---
    # ==============================================================================
    @testset "6.1 Settling Module Tests" begin
        
        @testset "Downward Movement and Flux Calculation" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            state.tracers[tracer_name][ng+1, ng+1, nz] = 1.0 # 1 kg/m^3 in top cell
            
            initial_mass = get_total_water_mass(state, grid, tracer_name)
            @test initial_mass ≈ 100.0 # 1 kg/m^3 * (10*10*1) m^3 volume

            dt = 10.0
            deposition_fluxes = apply_settling!(state, grid, dt, sediment_params)

            @test state.tracers[tracer_name][ng+1, ng+1, nz] < 1.0
            @test state.tracers[tracer_name][ng+1, ng+1, nz-1] > 0.0

            ws = sediment_params[tracer_name].ws
            new_bottom_conc = state.tracers[tracer_name][ng+1, ng+1, 1]
            @test deposition_fluxes[tracer_name][ng+1, ng+1] ≈ ws * new_bottom_conc
        end

        @testset "Mass Conservation in Water Column" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            state.tracers[tracer_name] .= 0.5 # Uniform concentration
            initial_mass = get_total_water_mass(state, grid, tracer_name)
            
            dt = 50.0
            deposition_fluxes = apply_settling!(state, grid, dt, sediment_params)
            final_mass_in_water = get_total_water_mass(state, grid, tracer_name)
            
            deposited_mass = deposition_fluxes[tracer_name][ng+1, ng+1] * dt * (dx*dy)
            @test initial_mass ≈ final_mass_in_water + deposited_mass rtol=1e-9
        end

        @testset "Stability with Large Timestep (CFL > 1)" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            state.tracers[tracer_name] .= 1.0
            dt = 200.0 # CFL = 0.01 * 200 / 1.0 = 2.0
            
            apply_settling!(state, grid, dt, sediment_params)

            @test !any(isnan, state.tracers[tracer_name])
            @test !any(isinf, state.tracers[tracer_name])
            @test all(state.tracers[tracer_name] .>= 0)
        end
    end

    # ==============================================================================
    # --- 6.2: Tests for BedExchangeModule.jl ---
    # ==============================================================================
    @testset "6.2 Bed Exchange Module Tests" begin

        @testset "Correct Deposition to Bed" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            state.bed_mass[tracer_name] .= 0.0
            
            mock_deposition_flux = 0.002
            deposition_fluxes = Dict(tracer_name => fill(mock_deposition_flux, (nx_tot, ny_tot)))
            
            dt = 100.0
            params_no_erosion = Dict(tracer_name => SedimentParams(ws=0.01, erosion_rate=0.0))
            bed_exchange!(state, grid, dt, deposition_fluxes, params_no_erosion)

            @test state.bed_mass[tracer_name][ng+1, ng+1] ≈ mock_deposition_flux * dt
        end

        @testset "Correct Erosion from Bed" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            initial_bed_mass_per_area = 0.5
            state.bed_mass[tracer_name] .= initial_bed_mass_per_area
            
            deposition_fluxes = Dict(tracer_name => zeros(Float64, (nx_tot, ny_tot)))
            dt = 100.0
            erosion_rate = sediment_params[tracer_name].erosion_rate
            bed_exchange!(state, grid, dt, deposition_fluxes, sediment_params)

            eroded_mass_per_area = erosion_rate * dt
            @test state.bed_mass[tracer_name][ng+1, ng+1] ≈ initial_bed_mass_per_area - eroded_mass_per_area

            bottom_cell_volume = grid.volume[ng+1, ng+1, 1]
            expected_conc_increase = (eroded_mass_per_area * (dx*dy)) / bottom_cell_volume
            @test state.tracers[tracer_name][ng+1, ng+1, 1] ≈ expected_conc_increase
        end

        @testset "Erosion is Limited by Bed Mass" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            dt = 100.0
            erosion_rate = sediment_params[tracer_name].erosion_rate
            initial_bed_mass_per_area = erosion_rate * dt * 0.5 # Half of potential erosion
            state.bed_mass[tracer_name] .= initial_bed_mass_per_area
            
            deposition_fluxes = Dict(tracer_name => zeros(Float64, (nx_tot, ny_tot)))
            bed_exchange!(state, grid, dt, deposition_fluxes, sediment_params)

            @test state.bed_mass[tracer_name][ng+1, ng+1] ≈ 0.0 atol=1e-12

            bottom_cell_volume = grid.volume[ng+1, ng+1, 1]
            expected_conc_increase = (initial_bed_mass_per_area * (dx*dy)) / bottom_cell_volume
            @test state.tracers[tracer_name][ng+1, ng+1, 1] ≈ expected_conc_increase
        end

        @testset "Conservation of Total Mass (Water + Bed)" begin
            state = initialize_state(grid, (tracer_name,); sediment_tracers=[tracer_name])
            state.tracers[tracer_name] .= 0.5
            state.bed_mass[tracer_name] .= 0.1
            
            initial_total_mass = get_total_water_mass(state, grid, tracer_name) + get_total_bed_mass(state, grid, tracer_name)
            
            dt = 50.0
            deposition = apply_settling!(state, grid, dt, sediment_params)
            bed_exchange!(state, grid, dt, deposition, sediment_params)
            
            final_total_mass = get_total_water_mass(state, grid, tracer_name) + get_total_bed_mass(state, grid, tracer_name)
            
            @test initial_total_mass ≈ final_total_mass rtol=1e-9
        end
    end
# test/monotonicity_curvilinear.jl

@testset "7. Advection Scheme Monotonicity (Curvilinear Grid)" begin
    @info "Running Testset 7: Advection Scheme Monotonicity on a Curvilinear Grid..."

    # --- 1. Grid Setup ---
    ng = 2
    nx, ny, nz = 50, 1, 1
    Lx, Ly, Lz = 100.0, 10.0, 10.0
    dx, dy = Lx / nx, Ly / ny
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    pm = ones(Float64, nx_tot, ny_tot) ./ dx
    pn = ones(Float64, nx_tot, ny_tot) ./ dy
    h = ones(Float64, nx_tot, ny_tot) .* Lz
    
    zeros_arr = zeros(Float64, nx_tot, ny_tot)
    trues_arr_rho = trues(nx_tot, ny_tot)
    trues_arr_u = trues(nx + 1 + 2*ng, ny + 2*ng)
    trues_arr_v = trues(nx + 2*ng, ny + 1 + 2*ng)

    z_w = collect(range(-Lz, 0, length=nz+1))
    
    volume = zeros(Float64, nx_tot, ny_tot, nz)
    face_area_x = zeros(Float64, nx_tot + 1, ny_tot, nz)
    face_area_y = zeros(Float64, nx_tot, ny_tot + 1, nz)

    for k in 1:nz
        dz = abs(z_w[k+1] - z_w[k])
        volume[:, :, k] .= dx * dy * dz
        face_area_x[:, :, k] .= dy * dz
        face_area_y[:, :, k] .= dx * dz
    end

    grid = CurvilinearGrid(ng, nx, ny, nz, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, 
                           z_w, pm, pn, zeros_arr, h,
                           trues_arr_rho, trues_arr_u, trues_arr_v,
                           face_area_x, face_area_y, volume)

    # --- 2. Initial Conditions: A "Square Wave" ---
    tracer_name = :SquareWave
    initial_state = initialize_state(grid, (tracer_name,))
    C = initial_state.tracers[tracer_name]
    C .= 0.0
    physical_region_x = (ng+1):(nx+ng)
    start_idx, end_idx = floor(Int, nx/2)-5, floor(Int, nx/2)+5
    C[physical_region_x[start_idx:end_idx], ng+1, 1] .= 1.0

    initial_max = maximum(C)
    initial_min = minimum(C)
    @test initial_max == 1.0
    @test initial_min == 0.0

    # --- 3. Simulation and Tests ---
    dt = 0.5
    time_range = 0.0:dt:50.0

    @testset "7.1 TVD Scheme Test" begin
        state = deepcopy(initial_state)
        state.u .= 1.0 
        state.v .= 0.0
        
        for time in time_range
            horizontal_transport!(state, grid, dt, :TVD, 0.0, BoundaryCondition[])
        end
        
        C_final = state.tracers[tracer_name]
        @test maximum(C_final) <= initial_max + 1e-9
        @test minimum(C_final) >= initial_min - 1e-9
    end

    @testset "7.2 UP3 Scheme Test (Potential for Overshoots)" begin
        state = deepcopy(initial_state)
        state.u .= 1.0
        state.v .= 0.0

        for time in time_range
             horizontal_transport!(state, grid, dt, :UP3, 0.0, BoundaryCondition[])
        end

        C_final = state.tracers[tracer_name]
        @test maximum(C_final) < 1.1 
        @test minimum(C_final) > -0.1
    end

    @testset "7.3 Implicit ADI Scheme Test" begin
        state = deepcopy(initial_state)
        state.u .= 1.0
        state.v .= 0.0

        for time in time_range
             horizontal_transport!(state, grid, dt, :ImplicitADI, 0.0, BoundaryCondition[])
        end

        C_final = state.tracers[tracer_name]

        # The first-order implicit scheme is monotonic and should not create overshoots.
        @test maximum(C_final) <= initial_max + 1e-9
        @test minimum(C_final) >= initial_min - 1e-9
    end
end
end