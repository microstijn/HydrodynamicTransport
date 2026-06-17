# test/benchmarks/advection_benchmarks.jl
#
# Group A — horizontal advection benchmarks (schemes :FFSL, :TVD, :UP3) against analytical flows.
# Pure advection is isolated through the real run_simulation path with Kh = 0 (the revived kwarg),
# a single layer (nz = 1, no vertical transport) and a fixed dt. Cases:
#   A1  uniform translation of a Gaussian  -> order-of-accuracy (grid refinement)
#   A2  solid-body rotation of a Gaussian  -> L1/L2/L∞, peak retention, mass drift (1 revolution)
#   A3  Zalesak slotted cylinder (rotation)-> monotonicity: under/overshoot, slot fidelity
#
# These define plain functions (no top-level work) so both validate_advection.jl and runtests.jl
# can call them. Requires BenchmarkCommon to be in scope (include benchmark_common.jl first).

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using NCDatasets
# `quiet` comes from BenchmarkCommon (mutes stdout+stderr around run_simulation).

"""
    advect_run(scheme, ufun, vfun, icfun, exactfun; nx, ny, dx, depth, tmax, nsteps)

Run one horizontal-advection case and return a NamedTuple of metrics. Velocity is the analytic
(ufun,vfun); the tracer is initialised from `icfun(x,y,z)` and compared to `exactfun(x,y,z)` at the
final time. Pure advection: Kh = 0, nz = 1, fixed dt = tmax/nsteps, FFSL/TVD/UP3 per `scheme`.
"""
function advect_run(scheme::Symbol, ufun, vfun, icfun, exactfun;
                    nx::Int, ny::Int, dx::Float64, depth::Float64=10.0,
                    tmax::Float64, nsteps::Int)
    dt = tmax / nsteps
    mktempdir() do dir
        path = joinpath(dir, "vel.nc")
        write_velocity_nc(path; nx=nx, ny=ny, nz=1, dx=dx, dy=dx, depth=depth,
                          ufun=ufun, vfun=vfun, tmax=tmax * 10)
        local norms, peak_ret, C0peak
        quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            set_tracer!(state, grid, icfun)
            C0peak = maximum(phys_view(state.tracers[:C], grid))
            final = run_simulation(grid, state, PointSource[], 0.0, tmax, dt;
                                   ds=ds, hydro_data=hydro, advection_scheme=scheme,
                                   use_adaptive_dt=false, Kh=0.0,
                                   diagnose_vertical_velocity=false, write_full_state=false)
            close(ds)
            # exact field on physical cells
            exact = similar(Array(phys_view(final.tracers[:C], grid)), Float64)
            ng = grid.ng
            for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
                exact[i, j, k] = exactfun(xcenter(i, dx), ycenter(j, dx), zcenter(k, grid.nz))
            end
            Cnum = Array(phys_view(final.tracers[:C], grid))
            vol = Array(phys_view(grid.volume, grid))
            norms = error_norms(Cnum, exact, vol)
            peak_ret = norms.maxval / C0peak
        end
        return (scheme=scheme, nx=nx, dx=dx, dt=dt, nsteps=nsteps,
                L1=norms.L1, L2=norms.L2, Linf=norms.Linf,
                mass_drift=norms.mass_drift, minval=norms.minval, maxval=norms.maxval,
                peak_retention=peak_ret)
    end
end

# --- A1: uniform translation, order-of-accuracy over a grid-refinement sweep ---
"""
    bench_translation(scheme; resolutions, U, courant) -> Vector of per-resolution metrics

Advect a Gaussian hill a fixed physical distance by a constant velocity U (in +x), refining the grid.
The domain/time are fixed in physical units; dx shrinks with resolution and dt is set to hold the
advective Courant `courant` fixed. Exact solution = shifted Gaussian.
"""
function bench_translation(scheme::Symbol; resolutions=[25, 50, 100, 200],
                           U::Float64=1.0, courant::Float64=0.5, L::Float64=100.0)
    σ = L / 16; xc0 = L / 4; yc = L / 2
    travel = L / 2                      # final centre at 3L/4 -> stays interior
    tmax = travel / U
    rows = NamedTuple[]
    for nres in resolutions
        dx = L / nres
        dt_target = courant * dx / U
        nsteps = max(1, round(Int, tmax / dt_target))
        ufun = (x, y) -> U; vfun = (x, y) -> 0.0
        ic = gaussian_hill(xc0, yc, σ)
        exact = gaussian_hill(xc0 + U * tmax, yc, σ)
        push!(rows, advect_run(scheme, ufun, vfun, ic, exact;
                               nx=nres, ny=nres, dx=dx, tmax=tmax, nsteps=nsteps))
    end
    return rows
end

# --- A2 / A3: solid-body rotation (one full revolution -> exact = initial condition) ---
"""
    bench_rotation(scheme, icfun; nx, revolutions, courant) -> metrics

Solid-body rotation about the domain centre; after `revolutions` full turns the exact field equals
the initial condition. `courant` sets dt from the maximum (corner) speed; FFSL tolerates Courant > 1.
"""
function bench_rotation(scheme::Symbol, icfun; nx::Int=100, revolutions::Float64=1.0,
                        courant::Float64=0.5, L::Float64=100.0, Ω::Float64=2π / 100.0)
    dx = L / nx; cx = L / 2; cy = L / 2
    Rmax = hypot(L, L) / 2                 # farthest corner from centre
    Vmax = Ω * Rmax
    T = revolutions * 2π / Ω
    dt_target = courant * dx / Vmax
    nsteps = max(1, round(Int, T / dt_target))
    ufun = (x, y) -> -Ω * (y - cy)
    vfun = (x, y) ->  Ω * (x - cx)
    return advect_run(scheme, ufun, vfun, icfun, icfun;
                      nx=nx, ny=nx, dx=dx, tmax=T, nsteps=nsteps)
end

bench_gaussian_rotation(scheme; kw...) =
    bench_rotation(scheme, gaussian_hill(75.0, 50.0, 100.0 / 16); kw...)

bench_zalesak(scheme; kw...) =
    bench_rotation(scheme, zalesak_slotted_cylinder(75.0, 50.0, 100.0 / 8); kw...)
