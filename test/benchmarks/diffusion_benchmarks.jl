# test/benchmarks/diffusion_benchmarks.jl
#
# Group C — diffusion benchmarks against the analytical heat-equation (Gaussian-spreading) solution.
# Advection is switched off (zero velocity), so the operators are isolated:
#   C1  2-D horizontal diffusion : explicit central diffusion; a Gaussian of variance σ₀² spreads as
#       σ²(t) = σ₀² + 2·Kh·t. Order study over grid refinement -> p ≈ 2; plus a recovered-variance
#       check and mass conservation.
#   C2  1-D vertical CN diffusion : the per-column Crank-Nicolson solve, same Gaussian-spreading law
#       in the unit column. Order study over nz -> p ≈ 2.
#   C3  vertical CN stability     : a large diffusive Courant must stay finite and bounded (CN is
#       unconditionally stable).
#
# C1 uses the real run_simulation path with the revived Kh kwarg; C2/C3 drive vertical_transport!
# (w = 0 -> pure CN diffusion) with the revived Kz kwarg.

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using NCDatasets

# 2-D mass-conserving Gaussian: amplitude scales as σ₀²/σ² (∫ = const).
_gauss2d(x, y, xc, yc, σ, amp0, σ0) = amp0 * (σ0^2 / σ^2) * exp(-((x - xc)^2 + (y - yc)^2) / (2σ^2))
# 1-D mass-conserving Gaussian: amplitude scales as σ₀/σ.
_gauss1d(z, zc, σ, amp0, σ0) = amp0 * (σ0 / σ) * exp(-(z - zc)^2 / (2σ^2))

# --- C1: 2-D horizontal diffusion order study ---
"""
    bench_horizontal_diffusion(; resolutions, Kh, L, tmax, σ0, diffnum) -> per-resolution metrics

Spread a Gaussian by horizontal diffusion `Kh`; exact = Gaussian with σ²(t)=σ₀²+2·Kh·t. Grid is
refined at a fixed diffusion number (dt ∝ dx²) so the measured order reflects the spatial scheme.
Returns L2, recovered variance, σ²-error and mass drift per resolution.
"""
function bench_horizontal_diffusion(; resolutions=[40, 80, 160], Kh::Float64=2.0, L::Float64=160.0,
                                    tmax::Float64=40.0, σ0::Float64=8.0, diffnum::Float64=0.2)
    xc = L / 2; yc = L / 2
    σend = sqrt(σ0^2 + 2Kh * tmax)
    rows = NamedTuple[]
    for nres in resolutions
        dx = L / nres
        dt = diffnum * dx^2 / Kh
        nsteps = max(1, round(Int, tmax / dt)); dt = tmax / nsteps
        row = mktempdir() do dir
            path = joinpath(dir, "still.nc")
            write_velocity_nc(path; nx=nres, ny=nres, nz=1, dx=dx, dy=dx, depth=10.0,
                              ufun=(x, y) -> 0.0, vfun=(x, y) -> 0.0, tmax=tmax * 10)
            quiet() do
                grid, ds, hydro, state = build_grid_state(path)
                set_tracer!(state, grid, (x, y, z) -> _gauss2d(x, y, xc, yc, σ0, 1.0, σ0))
                final = run_simulation(grid, state, PointSource[], 0.0, tmax, dt;
                                       ds=ds, hydro_data=hydro, advection_scheme=:FFSL,
                                       use_adaptive_dt=false, Kh=Kh,
                                       diagnose_vertical_velocity=false, write_full_state=false)
                close(ds)
                Cnum = Array(phys_view(final.tracers[:C], grid))
                vol = Array(phys_view(grid.volume, grid))
                exact = similar(Cnum, Float64)
                for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
                    exact[i, j, k] = _gauss2d(xcenter(i, dx), ycenter(j, dx), xc, yc, σend, 1.0, σ0)
                end
                en = error_norms(Cnum, exact, vol)
                # recovered variance: E[r²] = 2σ² for a 2-D Gaussian
                m0 = 0.0; mr = 0.0
                for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
                    c = Float64(Cnum[i, j, k]); r2 = (xcenter(i, dx) - xc)^2 + (ycenter(j, dx) - yc)^2
                    m0 += c; mr += c * r2
                end
                σ2_num = mr / m0 / 2
                (nx=nres, dx=dx, dt=dt, nsteps=nsteps, L2=en.L2, Linf=en.Linf,
                 mass_drift=en.mass_drift, sigma2_num=σ2_num, sigma2_exact=σend^2,
                 sigma2_relerr=abs(σ2_num - σend^2) / σend^2)
            end
        end
        push!(rows, row)
    end
    return rows
end

# --- shared vertical CN driver (w = 0 -> pure diffusion) ---
function _run_vertical_diffusion(; nz::Int, Kz::Float64, dt::Float64, nsteps::Int, icfun)
    mktempdir() do dir
        path = joinpath(dir, "still.nc")
        write_velocity_nc(path; nx=3, ny=3, nz=nz, dx=1.0, dy=1.0, depth=10.0,
                          ufun=(x, y) -> 0.0, vfun=(x, y) -> 0.0, tmax=dt * nsteps * 10)
        quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            close(ds)
            ng = grid.ng
            for k in 1:nz, j in 1:grid.ny, i in 1:grid.nx
                state.tracers[:C][i+ng, j+ng, k] = icfun(zcenter(k, nz))
            end
            fill!(state.w, 0.0)                                   # diffusion only
            for _ in 1:nsteps
                vertical_transport!(state, grid, dt; Kz=Kz)
            end
            ig = ng + 2; jg = ng + 2
            Float64[state.tracers[:C][ig, jg, k] for k in 1:nz]
        end
    end
end

# --- C2: 1-D vertical CN diffusion order study ---
"""
    bench_vertical_diffusion(; resolutions, Kz, tmax, σ0, diffnum) -> per-resolution metrics

Spread a Gaussian in the unit column by CN vertical diffusion; exact = Gaussian σ²=σ₀²+2·Kz·t (kept
narrow/interior so reflective-boundary images are negligible). Refines nz at fixed diffusion number.
"""
function bench_vertical_diffusion(; resolutions=[20, 40, 80, 160], Kz::Float64=1e-3,
                                  tmax::Float64=4.0, σ0::Float64=0.08, diffnum::Float64=0.25)
    zc = 0.5
    σend = sqrt(σ0^2 + 2Kz * tmax)
    rows = NamedTuple[]
    for nz in resolutions
        dz = 1 / nz
        dt = diffnum * dz^2 / Kz
        nsteps = max(1, round(Int, tmax / dt)); dt = tmax / nsteps
        ic = z -> _gauss1d(z, zc, σ0, 1.0, σ0)
        col = _run_vertical_diffusion(; nz=nz, Kz=Kz, dt=dt, nsteps=nsteps, icfun=ic)
        exact = [_gauss1d(zcenter(k, nz), zc, σend, 1.0, σ0) for k in 1:nz]
        d = abs.(col .- exact)
        L2 = sqrt(sum(d .^ 2) / nz)
        # recovered variance E[(z-zc)²] = σ²
        m0 = sum(col); mz = sum(col[k] * (zcenter(k, nz) - zc)^2 for k in 1:nz)
        σ2_num = mz / m0
        push!(rows, (nz=nz, dz=dz, dt=dt, nsteps=nsteps, L2=L2,
                     sigma2_num=σ2_num, sigma2_exact=σend^2,
                     sigma2_relerr=abs(σ2_num - σend^2) / σend^2,
                     minval=minimum(col), maxval=maximum(col)))
    end
    return rows
end

# --- C3: vertical CN stability at large diffusive Courant ---
"""
    bench_vertical_diffusion_stability(; nz, diffnum, nsteps, Kz) -> (diffnum, minval, maxval, finite)

Run CN vertical diffusion at a large diffusion number (Kz·dt/dz² ≫ 1) and confirm the result stays
finite and bounded by the initial peak (CN is unconditionally stable).
"""
function bench_vertical_diffusion_stability(; nz::Int=40, diffnum::Float64=10.0, nsteps::Int=50,
                                            Kz::Float64=1e-3)
    dz = 1 / nz; dt = diffnum * dz^2 / Kz
    ic = z -> exp(-(z - 0.5)^2 / (2 * 0.08^2))
    col = _run_vertical_diffusion(; nz=nz, Kz=Kz, dt=dt, nsteps=nsteps, icfun=ic)
    return (diffnum=diffnum, minval=minimum(col), maxval=maximum(col), finite=all(isfinite, col))
end
