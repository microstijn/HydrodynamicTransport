# test/benchmarks/vertical_benchmarks.jl
#
# Group B — vertical-transport benchmarks: the diagnosed-omega + implicit upwind/CN path, which is
# the least-validated part of the solver. Cases:
#   B1  constancy preservation : a depth-integral-zero, z-dependent horizontal strain field has a
#       nonzero continuity vertical velocity; with the diagnosed w a UNIFORM tracer must stay uniform
#       (the design goal of diagnose_vertical_velocity!). Compared against w ≡ 0, which does not.
#   B2  1-D vertical advection : constant interior w advects a Gaussian; exact = shifted profile.
#       Implicit upwind is 1st-order -> order study over nz refinement gives p ≈ 1.
#   B3  high-Courant stability : vertical Courant ≫ 1 must stay finite, positive and bounded
#       (the rationale for the unconditionally-stable implicit solve).
#
# B1 runs through the real run_simulation path; B2/B3 drive the vertical_transport! kernel directly
# (constant w cannot be produced by the rigid-lid continuity diagnosis, so it is injected).

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.VerticalTransportModule: vertical_transport!
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets

# --- B1: continuity-closure of the diagnosed vertical velocity ---
"""
    bench_continuity_closure(; nx, nz) -> (residual_diagnosed, residual_zero_w)

Validate `diagnose_vertical_velocity!` by the physical property it must satisfy: with the diagnosed
w the discrete net volume-flux divergence of every interior cell must vanish for a depth-integrated
non-divergent (Σ_k HDiv = 0) flow. We impose a boundary-vanishing, z-dependent sinusoidal field,
diagnose w, then independently compute each cell's net outflow `HDiv(k) + (w[k+1]−w[k])·area` from
state.u/v/w and the grid face areas, and return the max |residual| relative to the face-flux scale.
Diagnosed → ~machine zero (closure holds); the same metric with w ≡ 0 is O(1) (the uncompensated
horizontal divergence), showing the diagnosis is both correct and necessary.
"""
function bench_continuity_closure(; nx::Int=20, nz::Int=8)
    dx = 1.0; Lx = nx * dx
    gk = 0.1 .* ([Float64(k) - (nz + 1) / 2 for k in 1:nz]) ./ (nz / 2)   # Σ gk = 0
    ufun = (x, y) -> sin(2π * x / Lx)
    mktempdir() do dir
        path = joinpath(dir, "strain.nc")
        write_velocity_nc(path; nx=nx, ny=nx, nz=nz, dx=dx, dy=dx, depth=10.0,
                          ufun=ufun, vfun=(x, y) -> 0.0, tmax=10.0,
                          uprofile=gk, vprofile=zeros(nz))
        return quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            update_hydrodynamics!(state, grid, ds, hydro, 0.0; diagnose_w=true)
            close(ds)
            ng = grid.ng; u = state.u; v = state.v; w = state.w
            fax = grid.face_area_x; fay = grid.face_area_y
            maxres_diag = 0.0; maxres_zero = 0.0; fluxscale = 0.0
            for k in 1:nz, j in 2:grid.ny-1, i in 2:grid.nx-1     # interior diagnosable cells
                ig = i + ng; jg = j + ng
                area = 1.0 / (grid.pm[ig, jg] * grid.pn[ig, jg])
                hdiv = u[ig+1, jg, k] * fax[ig+1, jg, k] - u[ig, jg, k] * fax[ig, jg, k] +
                       v[ig, jg+1, k] * fay[ig, jg+1, k] - v[ig, jg, k] * fay[ig, jg, k]
                vdiv = (w[ig, jg, k+1] - w[ig, jg, k]) * area
                maxres_diag = max(maxres_diag, abs(hdiv + vdiv))
                maxres_zero = max(maxres_zero, abs(hdiv))             # w ≡ 0 -> residual is HDiv
                fluxscale = max(fluxscale, abs(u[ig, jg, k] * fax[ig, jg, k]))
            end
            sc = fluxscale > 0 ? fluxscale : 1.0
            (residual_diagnosed=maxres_diag / sc, residual_zero_w=maxres_zero / sc)
        end
    end
end

# --- B1b: the rigid-lid LIMITATION — barotropic (depth-integral-nonzero) divergence is NOT closed ---
"""
    bench_continuity_barotropic(; nx, nz) -> (residual_diagnosed, residual_zero_w)

Companion to `bench_continuity_closure` that exercises the case it deliberately excludes: a **barotropic**
(depth-uniform) horizontal strain, for which the column-integrated divergence `Σ_k HDiv ≠ 0` (`Dtot ≠ 0`).
The rigid-lid diagnosis pins the surface flux to zero and distributes `Dtot` by layer thickness, so it can
only absorb the depth-integral-ZERO part; the barotropic part is left uncompensated. The net-flux
divergence residual therefore does NOT go to machine zero — it stays ≈ the raw horizontal divergence
(`residual_diagnosed ≈ residual_zero_w`, ratio ≈ 1). This documents, as a regression test, that a uniform
tracer is preserved only for depth-integrated non-divergent flow: the frozen-volume solver has no tidal
breathing. On the real macrotidal CurviLoire grid this manifests as a ~0.5 log10 C≡1 error at the
intertidal receptor; it is an offline velocity-snapshot reconstruction limit, not a solver bug fixable by
a free-surface refactor (that would need native mass fluxes). See `claude/SOLVER_VALIDATION.md`.
"""
function bench_continuity_barotropic(; nx::Int=20, nz::Int=8)
    dx = 1.0; Lx = nx * dx
    ufun = (x, y) -> sin(2π * x / Lx)                 # depth-uniform amplitude -> barotropic (Σ_k HDiv ≠ 0)
    mktempdir() do dir
        path = joinpath(dir, "strain_bt.nc")
        write_velocity_nc(path; nx=nx, ny=nx, nz=nz, dx=dx, dy=dx, depth=10.0,
                          ufun=ufun, vfun=(x, y) -> 0.0, tmax=10.0,
                          uprofile=ones(nz), vprofile=zeros(nz))
        return quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            update_hydrodynamics!(state, grid, ds, hydro, 0.0; diagnose_w=true)
            close(ds)
            ng = grid.ng; u = state.u; v = state.v; w = state.w
            fax = grid.face_area_x; fay = grid.face_area_y
            maxres_diag = 0.0; maxres_zero = 0.0; fluxscale = 0.0
            for k in 1:nz, j in 2:grid.ny-1, i in 2:grid.nx-1
                ig = i + ng; jg = j + ng
                area = 1.0 / (grid.pm[ig, jg] * grid.pn[ig, jg])
                hdiv = u[ig+1, jg, k] * fax[ig+1, jg, k] - u[ig, jg, k] * fax[ig, jg, k] +
                       v[ig, jg+1, k] * fay[ig, jg+1, k] - v[ig, jg, k] * fay[ig, jg, k]
                vdiv = (w[ig, jg, k+1] - w[ig, jg, k]) * area
                maxres_diag = max(maxres_diag, abs(hdiv + vdiv))
                maxres_zero = max(maxres_zero, abs(hdiv))
                fluxscale = max(fluxscale, abs(u[ig, jg, k] * fax[ig, jg, k]))
            end
            sc = fluxscale > 0 ? fluxscale : 1.0
            (residual_diagnosed=maxres_diag / sc, residual_zero_w=maxres_zero / sc)
        end
    end
end

# --- shared driver for B2/B3: inject constant interior w, step vertical_transport! directly ---
function _run_vertical_column(; nz::Int, W::Float64, Kz::Float64, dt::Float64, nsteps::Int,
                              icfun, surface_outflow::Bool=false)
    mktempdir() do dir
        path = joinpath(dir, "still.nc")
        write_velocity_nc(path; nx=3, ny=3, nz=nz, dx=1.0, dy=1.0, depth=10.0,
                          ufun=(x, y) -> 0.0, vfun=(x, y) -> 0.0, tmax=dt * nsteps * 10)
        col = quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            close(ds)
            ng = grid.ng
            # Gaussian (or arbitrary) profile in z, identical across columns.
            for k in 1:nz, j in 1:grid.ny, i in 1:grid.nx
                state.tracers[:C][i+ng, j+ng, k] = icfun(zcenter(k, nz))
            end
            # Constant w at interior bottom-faces (k=2..nz) of EVERY column (full padded array,
            # incl. ghosts); seabed (k=1) stays 0. The surface face (k=nz+1) is 0 by default (closed
            # top -> blob translates within the column) or W when `surface_outflow` (open top -> mass
            # leaves, so a long high-Courant run cannot pile up against the surface).
            fill!(state.w, 0.0)
            ktop = surface_outflow ? nz + 1 : nz
            for k in 2:ktop, j in axes(state.w, 2), i in axes(state.w, 1)
                state.w[i, j, k] = W
            end
            for _ in 1:nsteps
                vertical_transport!(state, grid, dt; Kz=Kz)
            end
            ig = ng + 2; jg = ng + 2                          # an interior column
            Float64[state.tracers[:C][ig, jg, k] for k in 1:nz]
        end
        return col
    end
end

# --- B2: 1-D vertical advection order study ---
"""
    bench_vertical_advection(; resolutions, W, courant) -> per-resolution metrics

Advect a Gaussian in z by a constant interior w; exact = shifted Gaussian. Refines nz at fixed
vertical Courant. Pure advection (Kz = 0). Returns L2 + min/max per resolution; expect p ≈ 1.
"""
function bench_vertical_advection(; resolutions=[40, 80, 160, 320], W::Float64=1.0,
                                  courant::Float64=0.4)
    zc0 = 0.45; travel = 0.1; σz = 0.1; tmax = travel / W
    rows = NamedTuple[]
    for nz in resolutions
        dz = 1 / nz
        dt = courant * dz / W
        nsteps = max(1, round(Int, tmax / dt))
        ic = z -> exp(-(z - zc0)^2 / (2σz^2))
        col = _run_vertical_column(; nz=nz, W=W, Kz=0.0, dt=dt, nsteps=nsteps, icfun=ic)
        zc1 = zc0 + W * (dt * nsteps)
        exact = [exp(-(zcenter(k, nz) - zc1)^2 / (2σz^2)) for k in 1:nz]
        d = abs.(col .- exact)
        L2 = sqrt(sum(d .^ 2) / nz)
        push!(rows, (nz=nz, dz=dz, dt=dt, nsteps=nsteps, L2=L2,
                     minval=minimum(col), maxval=maximum(col)))
    end
    return rows
end

# --- B3: high-Courant stability ---
"""
    bench_vertical_stability(; nz, courant, nsteps, W) -> (max_courant, minval, maxval, finite)

Drive the implicit vertical advection at a vertical Courant ≫ 1 and confirm the solution stays finite,
positive and bounded by the initial peak (no explicit-upwind blow-up).
"""
function bench_vertical_stability(; nz::Int=80, courant::Float64=5.0, nsteps::Int=60, W::Float64=1.0)
    dz = 1 / nz; dt = courant * dz / W
    ic = z -> exp(-(z - 0.3)^2 / (2 * 0.05^2))
    mktempdir() do dir
        path = joinpath(dir, "still.nc")
        write_velocity_nc(path; nx=3, ny=3, nz=nz, dx=1.0, dy=1.0, depth=10.0,
                          ufun=(x, y) -> 0.0, vfun=(x, y) -> 0.0, tmax=dt * nsteps * 10)
        return quiet() do
            grid, ds, hydro, state = build_grid_state(path); close(ds); ng = grid.ng
            for k in 1:nz, j in 1:grid.ny, i in 1:grid.nx
                state.tracers[:C][i+ng, j+ng, k] = ic(zcenter(k, nz))
            end
            # Open top (surface outflow) so the blob advects up and OUT — no closed-surface pile-up.
            fill!(state.w, 0.0)
            for k in 2:nz+1, j in axes(state.w, 2), i in axes(state.w, 1)
                state.w[i, j, k] = W
            end
            ig = ng + 2; jg = ng + 2
            runmax = -Inf; runmin = Inf; finite = true
            for _ in 1:nsteps
                vertical_transport!(state, grid, dt; Kz=0.0)
                col = @view state.tracers[:C][ig, jg, :]
                runmax = max(runmax, maximum(col)); runmin = min(runmin, minimum(col))
                finite &= all(isfinite, col)
            end
            # Per-step Courant ≫ 1: implicit solve must stay finite, positive and bounded by the
            # initial peak (1.0) across the WHOLE run (an explicit upwind would blow up here).
            (max_courant=courant, minval=runmin, maxval=runmax, finite=finite)
        end
    end
end
