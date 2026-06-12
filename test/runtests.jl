# test/runtests.jl
#
# Self-contained test suite for HydrodynamicTransport.jl.
#
# All fixtures are synthetic (uniform grids built in-memory, tiny NetCDF files written to a
# temp dir), so the suite needs no external data and runs in a few seconds. Run with:
#
#     julia --project=. -e 'using Pkg; Pkg.test()'

using Test
using HydrodynamicTransport
using NCDatasets

# Internal (non-exported) functions live in submodules; bring the ones we exercise into scope.
using HydrodynamicTransport.FluxLimitersModule: van_leer, minmod, superbee, mc, calculate_limited_flux
using HydrodynamicTransport.HorizontalTransportModule: horizontal_transport!
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.BoundaryConditionsModule: apply_boundary_conditions!
using HydrodynamicTransport.SourceSinkModule: source_sink_terms!
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!, update_hydrodynamics_placeholder!
using HydrodynamicTransport.SettlingModule: apply_settling!
using HydrodynamicTransport.BedExchangeModule: bed_exchange!
using HydrodynamicTransport.VectorOperationsModule: rotate_velocities_to_grid!, rotate_velocities_to_geographic

# ------------------------------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------------------------------

"""Build a uniform `CurvilinearGrid` directly (no NetCDF), all cells water."""
function uniform_curvi_grid(; nx, ny, nz, dx = 10.0, dy = 10.0, Lz = 10.0, ng = 2)
    nx_tot, ny_tot = nx + 2ng, ny + 2ng
    pm = fill(1 / dx, nx_tot, ny_tot)
    pn = fill(1 / dy, nx_tot, ny_tot)
    h = fill(Lz, nx_tot, ny_tot)
    zeros_arr = zeros(Float64, nx_tot, ny_tot)
    z_w = collect(range(-Lz, 0.0, length = nz + 1))
    volume = zeros(Float64, nx_tot, ny_tot, nz)
    face_area_x = zeros(Float64, nx_tot + 1, ny_tot, nz)
    face_area_y = zeros(Float64, nx_tot, ny_tot + 1, nz)
    for k in 1:nz
        dz = abs(z_w[k+1] - z_w[k])
        volume[:, :, k] .= dx * dy * dz
        face_area_x[:, :, k] .= dy * dz
        face_area_y[:, :, k] .= dx * dz
    end
    return CurvilinearGrid(ng, nx, ny, nz,
        zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr,
        z_w, pm, pn, zeros_arr, h,
        trues(nx_tot, ny_tot), trues(nx + 1 + 2ng, ny + 2ng), trues(nx + 2ng, ny + 1 + 2ng),
        face_area_x, face_area_y, volume)
end

"""Write a tiny ROMS-like NetCDF (grid + time-varying u/v) usable for grid init and forcing."""
function write_synthetic_nc(path; nx = 8, ny = 6, nz = 3, ntime = 2,
                            u_val = 0.1, v_val = 0.05, land_corner = true)
    ds = NCDataset(path, "c")
    for (d, n) in (("xi_rho", nx), ("eta_rho", ny), ("xi_u", nx), ("eta_u", ny),
                   ("xi_v", nx), ("eta_v", ny), ("s_rho", nz), ("ocean_time", ntime))
        defDim(ds, d, n)
    end
    lon = [-2.0 + 0.01 * (i - 1) for i in 1:nx, j in 1:ny]
    lat = [47.0 + 0.01 * (j - 1) for i in 1:nx, j in 1:ny]
    defVar(ds, "lon_rho", lon, ("xi_rho", "eta_rho"))
    defVar(ds, "lat_rho", lat, ("xi_rho", "eta_rho"))
    # Zero the corner depth so the land cell AND its u/v faces are inferred consistently as
    # land by initialize_curvilinear_grid (mask_u/mask_v are not supplied, hence inferred from h).
    h = fill(10.0, nx, ny)
    mask = ones(Int, nx, ny)
    if land_corner
        h[1, 1] = 0.0
        mask[1, 1] = 0
    end
    defVar(ds, "h", h, ("xi_rho", "eta_rho"))
    defVar(ds, "dx", fill(100.0, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "dy", fill(100.0, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "angle", zeros(nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "mask_rho", mask, ("xi_rho", "eta_rho"))
    defVar(ds, "ocean_time", collect(range(0.0, 3600.0, length = ntime)), ("ocean_time",))
    u = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))
    v = defVar(ds, "v", Float64, ("xi_v", "eta_v", "s_rho", "ocean_time"))
    for t in 1:ntime
        u[:, :, :, t] = fill(u_val, nx, ny, nz)
        v[:, :, :, t] = fill(v_val, nx, ny, nz)
    end
    close(ds)
    return path
end

# ------------------------------------------------------------------------------------------

@testset "HydrodynamicTransport.jl" begin

    @testset "Flux limiters" begin
        # phi(1) == 1 for the symmetric limiters; phi(r<0) == 0 (TVD region).
        for f in (van_leer, minmod, mc)
            @test f(1.0) ≈ 1.0
            @test f(-2.0) == 0.0
        end
        @test superbee(1.0) ≈ 1.0
        # Superbee stays within the TVD bounds [0, 2].
        for r in range(-1.0, 4.0, length = 21)
            @test 0.0 <= superbee(r) <= 2.0
        end
        # In a flat region (zero gradient) the limited flux reduces to plain upwind advection.
        c, vel, area = 3.0, 2.0, 5.0
        @test calculate_limited_flux(c, c, c, vel, area, van_leer) ≈ vel * c * area
    end

    @testset "lonlat_to_ij" begin
        ng, nx, ny = 2, 10, 10
        nx_tot, ny_tot = nx + 2ng, ny + 2ng
        lon_rho = zeros(nx_tot, ny_tot); lat_rho = zeros(nx_tot, ny_tot)
        for j in 1:ny, i in 1:nx
            lon_rho[i+ng, j+ng] = Float64(i)
            lat_rho[i+ng, j+ng] = Float64(j)
        end
        mask_rho = trues(nx_tot, ny_tot)
        grid = CurvilinearGrid(ng, nx, ny, 1,
            lon_rho, lat_rho, zeros(nx-1+2ng, ny+2ng), zeros(nx-1+2ng, ny+2ng),
            zeros(nx+2ng, ny-1+2ng), zeros(nx+2ng, ny-1+2ng),
            [-1.0, 0.0], ones(nx_tot, ny_tot), ones(nx_tot, ny_tot), zeros(nx_tot, ny_tot), ones(nx_tot, ny_tot),
            mask_rho, trues(nx-1+2ng, ny+2ng), trues(nx+2ng, ny-1+2ng),
            zeros(nx_tot+1, ny_tot, 1), zeros(nx_tot, ny_tot+1, 1), zeros(nx_tot, ny_tot, 1))

        # Queries must lie within the water-cell bounding box [1,10] x [1,10].
        @test lonlat_to_ij(grid, 1.4, 1.4) == (1, 1)
        @test lonlat_to_ij(grid, 9.6, 9.6) == (10, 10)
        @test lonlat_to_ij(grid, 5.4, 6.2) == (5, 6)
        @test lonlat_to_ij(grid, 100.0, 100.0) === nothing   # out of bounds
        # Equidistant target -> first match + warning.
        ij = @test_warn "equidistant" lonlat_to_ij(grid, 3.5, 4.0)
        @test ij == (3, 4)
        # Land cell -> falls back to a neighbouring water cell.
        grid.mask_rho[3+ng, 3+ng] = false
        @test lonlat_to_ij(grid, 3.1, 3.1) in [(2, 3), (3, 2), (4, 3), (3, 4)]
        grid.mask_rho[3+ng, 3+ng] = true
    end

    @testset "State initialisation" begin
        grid = uniform_curvi_grid(nx = 4, ny = 4, nz = 3)
        state = initialize_state(grid, (:A, :B); sediment_tracers = [:B])
        for t in (:A, :B)
            @test haskey(state._buffer1, t) && haskey(state._buffer2, t)
            @test size(state._buffer1[t]) == size(state.tracers[t])
            @test size(state._buffer2[t]) == size(state.tracers[t])
        end
        @test haskey(state.bed_mass, :B)        # bed_mass only for sediment tracers
        @test !haskey(state.bed_mass, :A)
    end

    @testset "Cartesian grid geometry" begin
        nx, ny, nz, Lx, Ly, Lz = 20, 10, 5, 100.0, 50.0, 10.0
        grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
        dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
        @test grid.dims == [nx, ny, nz]
        @test grid.volume[grid.ng+1, grid.ng+1, 1] ≈ dx * dy * dz
        @test grid.face_area_x[grid.ng+2, grid.ng+1, 1] ≈ dy * dz
        @test grid.face_area_y[grid.ng+1, grid.ng+2, 1] ≈ dx * dz
    end

    @testset "Curvilinear grid + autodetect from NetCDF" begin
        mktempdir() do dir
            path = write_synthetic_nc(joinpath(dir, "grid.nc"))
            grid = initialize_curvilinear_grid(path)
            @test grid.nx == 8 && grid.ny == 6 && grid.nz == 3
            @test !grid.mask_rho[grid.ng+1, grid.ng+1]   # land corner (1,1)
            @test grid.mask_rho[grid.ng+4, grid.ng+3]    # interior water
            @test grid.pm[grid.ng+1, grid.ng+1] ≈ 1 / 100.0  # dx=100 -> pm=1/100

            hydro = create_hydrodynamic_data_from_file(path)
            @test hydro.var_map[:u] == "u" && hydro.var_map[:v] == "v"
            @test estimate_stable_timestep(hydro) > 0.0
        end
    end

    @testset "Hydrodynamics interpolation + slab cache" begin
        mktempdir() do dir
            path = joinpath(dir, "hydro.nc")
            ds = NCDataset(path, "c")
            defDim(ds, "xi_u", 1); defDim(ds, "eta_u", 1)
            defDim(ds, "s_rho", 1); defDim(ds, "ocean_time", 2)
            defVar(ds, "ocean_time", [0.0, 10.0], ("ocean_time",))
            uv = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))
            uv[:, :, :, 1] = fill(1.0, 1, 1, 1)
            uv[:, :, :, 2] = fill(3.0, 1, 1, 1)
            close(ds)

            grid = uniform_curvi_grid(nx = 1, ny = 1, nz = 1)
            state = initialize_state(grid, ())
            hydro = HydrodynamicData(path, Dict(:u => "u", :time => "ocean_time"))
            ds = NCDataset(path)
            gi = grid.ng + 1

            probe(t) = (update_hydrodynamics!(state, grid, ds, hydro, t); state.u[gi, gi, 1])
            @test probe(5.0) ≈ 2.0       # midpoint interpolation
            @test probe(-1.0) ≈ 1.0      # clamp below range
            @test probe(11.0) ≈ 3.0      # clamp above range
            @test probe(5.0) ≈ 2.0       # re-query (non-monotonic) hits the cache, same answer
            @test haskey(hydro.cache.slabs, 1)   # slab cache populated
            close(ds)
        end
    end

    @testset "Vector rotation round-trip" begin
        nx, ny, nz, ng = 6, 6, 1, 2
        grid = uniform_curvi_grid(nx = nx, ny = ny, nz = nz)
        # Constant 45-degree grid rotation.
        grid.angle .= π / 4
        state = initialize_state(grid, ())
        u_east = ones(Float64, nx, ny, nz); v_north = zeros(Float64, nx, ny, nz)
        rotate_velocities_to_grid!(state.u, state.v, grid, u_east, v_north)
        ue, vn = rotate_velocities_to_geographic(grid, state.u, state.v)
        @test ue ≈ u_east
        @test vn ≈ v_north atol = 1e-12
    end

    @testset "Core physics + mass bounds (Cartesian)" begin
        grid = initialize_cartesian_grid(20, 20, 5, 100.0, 100.0, 10.0)
        state = initialize_state(grid, (:C,))
        sources = [PointSource(i = 5, j = 10, k = 1, tracer_name = :C, influx_rate = t -> 100.0)]
        bcs = [OpenBoundary(side = :East), OpenBoundary(side = :West)]
        @test sum(state.tracers[:C] .* grid.volume) == 0.0

        dt, nsteps = 0.5, 10
        for _ in 1:nsteps
            state.time += dt
            apply_boundary_conditions!(state, grid, bcs)
            update_hydrodynamics_placeholder!(state, grid, state.time)
            horizontal_transport!(state, grid, dt, :TVD, 0.0, bcs)
            vertical_transport!(state, grid, dt)
            source_sink_terms!(state, grid, sources, FunctionalInteraction[], state.time, dt, 0.0)
        end
        final_mass = sum(state.tracers[:C] .* grid.volume)
        @test !any(isnan, state.tracers[:C])
        @test 0.0 < final_mass <= 100.0 * dt * nsteps + 1e-6
    end

    @testset "End-to-end curvilinear run (TVD & UP3)" begin
        mktempdir() do dir
            path = write_synthetic_nc(joinpath(dir, "forced.nc"))
            grid = initialize_curvilinear_grid(path)
            hydro = create_hydrodynamic_data_from_file(path)
            sources = [PointSource(i = 4, j = 3, k = grid.nz, tracer_name = :T, influx_rate = t -> 1.0e3)]
            bcs = [OpenBoundary(side = :East), OpenBoundary(side = :West),
                   OpenBoundary(side = :North), OpenBoundary(side = :South)]
            land_gi, land_gj = grid.ng + 1, grid.ng + 1   # physical (1,1) is land
            for scheme in (:TVD, :UP3)
                ds = NCDataset(path)
                state = initialize_state(grid, ds, (:T,))
                final = run_simulation(grid, state, sources, 0.0, 600.0, 60.0;
                                       ds = ds, hydro_data = hydro,
                                       boundary_conditions = bcs, advection_scheme = scheme)
                @test !any(isnan, final.tracers[:T])
                @test sum(final.tracers[:T]) > 0.0                       # source injected mass
                # Land cell stays effectively dry (only a negligible diffusive trace leaks in).
                @test maximum(abs, final.tracers[:T][land_gi, land_gj, :]) < 1e-4 * maximum(final.tracers[:T])
                close(ds)
            end
        end
    end

    @testset "Adaptive timestep" begin
        mktempdir() do dir
            path = write_synthetic_nc(joinpath(dir, "forced.nc"))
            grid = initialize_curvilinear_grid(path)
            hydro = create_hydrodynamic_data_from_file(path)
            ds = NCDataset(path)
            state = initialize_state(grid, ds, (:T,))
            sources = [PointSource(i = 4, j = 3, k = grid.nz, tracer_name = :T, influx_rate = t -> 1.0e3)]
            final = run_simulation(grid, state, sources, 0.0, 600.0, 60.0;
                                   ds = ds, hydro_data = hydro,
                                   use_adaptive_dt = true, cfl_max = 0.8,
                                   dt_max = 120.0, dt_min = 1.0)
            @test !any(isnan, final.tracers[:T])
            @test final.time >= 600.0 - 1e-6
            close(ds)
        end
    end

    @testset "Source/sink + functional decay" begin
        grid = uniform_curvi_grid(nx = 6, ny = 6, nz = 1, dx = 20.0, dy = 20.0)
        state = initialize_state(grid, (:C,))
        ng = grid.ng
        state.tracers[:C][ng+3, ng+3, 1] = 100.0

        # Point source adds rate*dt of mass to its cell.
        src = [PointSource(i = 2, j = 2, k = 1, tracer_name = :C, influx_rate = t -> 50.0)]
        vol = grid.volume[ng+2, ng+2, 1]
        source_sink_terms!(state, grid, src, FunctionalInteraction[], 0.0, 2.0, 0.0)
        @test state.tracers[:C][ng+2, ng+2, 1] ≈ (50.0 * 2.0) / vol

        # First-order decay removes a known fraction per step.
        k = 0.1
        decay = FunctionalInteraction(affected_tracers = [:C],
                                      interaction_function = (c, env, dt) -> Dict(:C => -k * c[:C] * dt))
        before = state.tracers[:C][ng+3, ng+3, 1]
        source_sink_terms!(state, grid, PointSource[], [decay], 0.0, 1.0, 0.0)
        @test state.tracers[:C][ng+3, ng+3, 1] ≈ before * (1 - k * 1.0)
    end

    @testset "Sediment settling + bed exchange" begin
        grid = uniform_curvi_grid(nx = 1, ny = 1, nz = 10, dx = 10.0, dy = 10.0, Lz = 10.0)
        ng = grid.ng
        tracer = :Sand
        params = Dict(tracer => SedimentParams(ws = 0.01, erosion_rate = 1e-6))
        cell_area = 10.0 * 10.0
        water_mass(s) = sum(s.tracers[tracer] .* grid.volume)
        bed_mass(s) = sum(view(s.bed_mass[tracer], ng+1:1+ng, ng+1:1+ng)) * cell_area

        # Settling moves tracer downward and produces a deposition flux.
        state = initialize_state(grid, (tracer,); sediment_tracers = [tracer])
        state.tracers[tracer][ng+1, ng+1, grid.nz] = 1.0
        dep = apply_settling!(state, grid, 10.0, params)
        @test state.tracers[tracer][ng+1, ng+1, grid.nz] < 1.0
        @test state.tracers[tracer][ng+1, ng+1, grid.nz-1] > 0.0
        @test dep[tracer][ng+1, ng+1] ≈ params[tracer].ws * state.tracers[tracer][ng+1, ng+1, 1]

        # Total mass (water + bed) is conserved across settling + bed exchange.
        state = initialize_state(grid, (tracer,); sediment_tracers = [tracer])
        state.tracers[tracer] .= 0.5
        state.bed_mass[tracer] .= 0.1
        total0 = water_mass(state) + bed_mass(state)
        dep = apply_settling!(state, grid, 50.0, params)
        bed_exchange!(state, grid, 50.0, dep, params)
        @test water_mass(state) + bed_mass(state) ≈ total0 rtol = 1e-9
    end

    @testset "Receptor monitoring" begin
        mktempdir() do dir
            grid = uniform_curvi_grid(nx = 8, ny = 6, nz = 3, dx = 50.0, dy = 50.0)
            state = initialize_state(grid, (:A, :B))
            sources = [PointSource(i = 4, j = 3, k = grid.nz, tracer_name = :A, influx_rate = t -> 1.0e3)]
            outfile = joinpath(dir, "receptor.csv")
            monitor = ReceptorMonitor(
                receptor_id = "R1", i = 4, j = 3,
                i_array = 4 + grid.ng, j_array = 3 + grid.ng,
                tracer_names = [:A, :B],
                normalization_by_tracer = Dict(:A => 1.0, :B => 1.0),
                output_file = outfile, start_time_seconds = 0.0)
            run_simulation(grid, state, sources, 0.0, 300.0, 60.0;
                           write_full_state = false,
                           receptor_monitors = [monitor], receptor_monitor_interval = 60.0)
            @test isfile(outfile)
            lines = readlines(outfile)
            @test length(lines) >= 2                       # header + at least one data row
            @test occursin("kernel_value", first(lines))   # header present
        end
    end

end
