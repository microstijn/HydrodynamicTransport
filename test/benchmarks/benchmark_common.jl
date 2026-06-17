# test/benchmarks/benchmark_common.jl
#
# Shared scaffolding for the analytical transport-solver benchmarks (Groups A/B/C). These validate
# the numerical operators against cases whose exact behaviour is known — the standard way to validate
# an advection–diffusion solver — and complement the existing internal-consistency suite
# (runtests.jl, validate_sigma.jl). All fixtures are synthetic and in-memory; no external data.
#
# Conventions (uniform grid, spacing dx,dy; physical index i,j ∈ 1:nx,1:ny):
#   - rho cell centre (i,j)        -> x = (i-0.5)·dx,  y = (j-0.5)·dy
#   - u-face for NetCDF u[i,j]      -> x = (i-1)·dx,    y = (j-0.5)·dy   (west face of cell i)
#   - v-face for NetCDF v[i,j]      -> x = (i-0.5)·dx,  y = (j-1)·dy     (south face of cell j)
# update_hydrodynamics! maps NetCDF u[i,j] -> state.u[i+ng, j+ng] with no rotation (angle = 0).
#
# On the non-sigma grid this NetCDF builds, the vertical layer thickness is the dimensionless legacy
# value dz = 1/nz (volume = dx·dy/nz), i.e. a unit-height column z ∈ (0,1). The vertical benchmarks
# (Groups B/C) work in that unit column with z_k = (k-0.5)/nz.

module BenchmarkCommon

using NCDatasets
using Statistics: mean
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs

export xcenter, ycenter, zcenter, write_velocity_nc, build_grid_state, set_tracer!,
       gaussian_hill, zalesak_slotted_cylinder, error_norms, ErrorNorms, fit_order,
       write_results_csv, phys_view, quiet

"""Run `f()` with stdout and stderr (ProgressMeter, autodetect prints) muted. Exceptions still
propagate, so genuine failures surface after the block."""
quiet(f) = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        f()
    end
end

# --- coordinate helpers (uniform grid) ---
@inline xcenter(i, dx) = (i - 0.5) * dx
@inline ycenter(j, dy) = (j - 0.5) * dy
@inline zcenter(k, nz) = (k - 0.5) / nz          # unit-height column, dz = 1/nz

"""
    write_velocity_nc(path; nx, ny, nz, dx, dy, depth, ufun, vfun, tmax)

Write a ROMS/MARS3D-like NetCDF carrying an analytic, time-constant staggered velocity field. `ufun`
and `vfun` are `(x, y) -> velocity`, sampled at the staggered u-/v-face locations. Two identical time
slices [0, tmax] let `update_hydrodynamics!` interpolate to a constant field at any time. All cells
are water (no land), depth = `depth`, spacing dx,dy (-> pm = 1/dx).
"""
function write_velocity_nc(path; nx::Int, ny::Int, nz::Int, dx::Float64, dy::Float64,
                           depth::Float64, ufun, vfun, tmax::Float64,
                           uprofile::AbstractVector=ones(nz), vprofile::AbstractVector=ones(nz))
    ds = NCDataset(path, "c")
    for (d, n) in (("xi_rho", nx), ("eta_rho", ny), ("xi_u", nx), ("eta_u", ny),
                   ("xi_v", nx), ("eta_v", ny), ("s_rho", nz), ("ocean_time", 2))
        defDim(ds, d, n)
    end
    # lon/lat are only used by lonlat lookups (not transport); store x/y for completeness.
    lon = [xcenter(i, dx) for i in 1:nx, j in 1:ny]
    lat = [ycenter(j, dy) for i in 1:nx, j in 1:ny]
    defVar(ds, "lon_rho", lon, ("xi_rho", "eta_rho"))
    defVar(ds, "lat_rho", lat, ("xi_rho", "eta_rho"))
    defVar(ds, "h", fill(depth, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "dx", fill(dx, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "dy", fill(dy, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "angle", zeros(nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "mask_rho", ones(Int, nx, ny), ("xi_rho", "eta_rho"))
    defVar(ds, "ocean_time", [0.0, tmax], ("ocean_time",))
    uvar = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))
    vvar = defVar(ds, "v", Float64, ("xi_v", "eta_v", "s_rho", "ocean_time"))
    uarr = [ufun(xcenter(i, dx) - 0.5dx, ycenter(j, dy)) for i in 1:nx, j in 1:ny]  # u-face: x=(i-1)dx
    varr = [vfun(xcenter(i, dx), ycenter(j, dy) - 0.5dy) for i in 1:nx, j in 1:ny]  # v-face: y=(j-1)dy
    # Optional separable per-layer profile u(x,y,k) = ufun(x,y)·uprofile[k] (used by the vertical
    # constancy-preservation test, which needs a depth-dependent horizontal field).
    for t in 1:2, k in 1:nz
        uvar[:, :, k, t] = uarr .* uprofile[k]
        vvar[:, :, k, t] = varr .* vprofile[k]
    end
    close(ds)
    return path
end

"""
    build_grid_state(path; tracer=:C) -> (grid, ds, hydro, state)

Initialise the curvilinear grid, hydro handle, open dataset and a zeroed state for one tracer from a
velocity NetCDF written by `write_velocity_nc`. Caller is responsible for `close(ds)`.
"""
function build_grid_state(path; tracer::Symbol=:C)
    grid = initialize_curvilinear_grid(path)
    hydro = create_hydrodynamic_data_from_file(path)
    ds = NCDataset(path)
    state = initialize_state(grid, ds, (tracer,))
    return grid, ds, hydro, state
end

"""
    set_tracer!(state, grid, f; tracer=:C)

Fill the physical cells of `state.tracers[tracer]` with `f(x, y, z)` (cell-centre coordinates);
ghost cells are left at 0 (fine for blobs that are ~0 near the domain edge).
"""
function set_tracer!(state::State, grid::CurvilinearGrid, f; tracer::Symbol=:C)
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
    dx = 1.0 / grid.pm[ng+1, ng+1]; dy = 1.0 / grid.pn[ng+1, ng+1]
    C = state.tracers[tracer]
    fill!(C, 0)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        C[i+ng, j+ng, k] = f(xcenter(i, dx), ycenter(j, dy), zcenter(k, nz))
    end
    return state
end

"""Physical-cell view of a padded field (drops ghosts)."""
phys_view(A, grid) = view(A, grid.ng+1:grid.nx+grid.ng, grid.ng+1:grid.ny+grid.ng, :)

# --- initial-condition shapes ---
"""Gaussian hill of amplitude `amp`, 1/e-width `σ`, centred at (xc,yc) (z ignored)."""
gaussian_hill(xc, yc, σ; amp=1.0) = (x, y, z) -> amp * exp(-((x - xc)^2 + (y - yc)^2) / (2σ^2))

"""Zalesak slotted cylinder: disc radius R at (xc,yc), with a rectangular slot of half-width `sw`
and depth `sd` cut from the +y side. Amplitude 1 inside, 0 outside (the classic FCT benchmark)."""
function zalesak_slotted_cylinder(xc, yc, R; sw=R/6, sd=1.5R, amp=1.0)
    return function (x, y, z)
        r = hypot(x - xc, y - yc)
        r > R && return 0.0
        # slot: a notch opening from the top (y > yc), centred in x
        (abs(x - xc) <= sw && (y - yc) >= (R - sd)) && return 0.0
        return amp
    end
end

# --- error metrics ---
struct ErrorNorms
    L1::Float64; L2::Float64; Linf::Float64
    mass_num::Float64; mass_exact::Float64; mass_drift::Float64
    minval::Float64; maxval::Float64; peak::Float64
end

"""
    error_norms(Cnum, Cexact, vol) -> ErrorNorms

Volume-weighted L1/L2 (RMS) and L∞ error of `Cnum` vs `Cexact` over equal-shape physical arrays,
plus mass (Σ C·vol) of each, relative mass drift, and min/max/peak of `Cnum`.
"""
function error_norms(Cnum::AbstractArray, Cexact::AbstractArray, vol::AbstractArray)
    Cn = Float64.(Cnum); Ce = Float64.(Cexact); V = Float64.(vol)
    Vtot = sum(V)
    d = abs.(Cn .- Ce)
    L1 = sum(d .* V) / Vtot
    L2 = sqrt(sum((d .^ 2) .* V) / Vtot)
    Linf = maximum(d)
    mass_num = sum(Cn .* V); mass_exact = sum(Ce .* V)
    drift = mass_exact != 0 ? (mass_num - mass_exact) / mass_exact : mass_num
    return ErrorNorms(L1, L2, Linf, mass_num, mass_exact, drift,
                      minimum(Cn), maximum(Cn), maximum(Cn))
end

"""Least-squares convergence order: slope of log(err) vs log(h)."""
function fit_order(hs::AbstractVector, errs::AbstractVector)
    x = log.(Float64.(hs)); y = log.(Float64.(errs))
    n = length(x); x̄ = mean(x); ȳ = mean(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

"""Write a simple CSV (no CSV.jl dependency). `rows` is a vector of vectors."""
function write_results_csv(path::String, header::Vector{String}, rows::Vector)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(string.(r), ","))
        end
    end
    return path
end

end # module BenchmarkCommon
