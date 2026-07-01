# test/benchmarks/reciprocity_benchmark.jl
#
# Group D — source<->receptor RECIPROCITY of the discrete transport operator.
#
# For a LINEAR passive tracer the adjoint (backward) transport equals the same advection-diffusion
# equation integrated on the velocity-FLIPPED flow. Reciprocity then states: the concentration at a
# receptor B produced by a unit release at a source A equals the concentration at A produced by a unit
# release at B under REVERSED velocity. (In the volume-weighted inner product the cell-volume factors
# cancel for a unit-MASS release read out as concentration; on the uniform benchmark grid all volumes
# are equal, so the single-cell spike series can be compared directly.)
#
# This is the mathematical basis for computing a receptor "footprint" / adjoint kernel with ONE
# backward run instead of one forward run per source (softMode step #1). For a STEADY field,
# time-reversal is identical to velocity negation, so this validates the reciprocity PRINCIPLE with
# the EXISTING forward solver (two NetCDFs, U and -U) BEFORE any reverse-time solver flag exists.
#
# Expectation (and the diagnostic): diffusion (Crank-Nicolson, symmetric stencil) is self-adjoint ->
# reciprocity to ~machine zero; flux-form advection reversed-in-velocity equals the discrete adjoint
# only up to truncation error -> reciprocity at the ~% level. The measured advective mismatch is the
# Gate-1 threshold reused by softMode E7 (reciprocity check of the real forward kernel library).
#
# Plain functions (no top-level work), same convention as advection_benchmarks.jl; requires
# BenchmarkCommon in scope (include benchmark_common.jl first).

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using NCDatasets

# relative L2 mismatch of two equal-length series (guarded against a zero forward signal)
function _rel_l2(f::AbstractVector, g::AbstractVector)
    n = min(length(f), length(g)); f = f[1:n]; g = g[1:n]
    den = sqrt(sum(abs2, f))
    return den > 0 ? sqrt(sum(abs2, f .- g)) / den : sqrt(sum(abs2, f .- g))
end

"""
    _spike_series(scheme, ufun, vfun, src, rcpt; nx,ny,nz,dx,depth,Kh,Kz,tmax,dt,store) -> Vector

Release a unit concentration spike in cell `src` (i,j,k physical indices), integrate the steady
(ufun,vfun) field, and return the concentration at cell `rcpt` at each stored time.
"""
function _spike_series(scheme::Symbol, ufun, vfun, src::NTuple{3,Int}, rcpt::NTuple{3,Int};
                       nx::Int, ny::Int, nz::Int, dx::Float64, depth::Float64,
                       Kh::Float64, Kz::Float64, tmax::Float64, dt::Float64, store::Float64)
    return mktempdir() do dir
        path = joinpath(dir, "vel.nc")
        write_velocity_nc(path; nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, depth=depth,
                          ufun=ufun, vfun=vfun, tmax=tmax * 10)
        series = Float64[]
        quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            ng = grid.ng
            fill!(state.tracers[:C], 0.0)
            state.tracers[:C][src[1]+ng, src[2]+ng, src[3]] = 1.0
            # steady, non-divergent fields -> diagnosed omega = 0 (no vertical advection); Kz sets
            # vertical diffusion only. run_and_store_simulation has no diagnose_vertical_velocity kwarg.
            results, _ = run_and_store_simulation(grid, state, PointSource[], 0.0, tmax, dt, store;
                                   ds=ds, hydro_data=hydro, advection_scheme=scheme,
                                   use_adaptive_dt=false, Kh=Kh, Kz=Kz, write_full_state=true)
            close(ds)
            for r in results
                push!(series, r.state.tracers[:C][rcpt[1]+ng, rcpt[2]+ng, rcpt[3]])
            end
        end
        return series
    end
end

"""
    bench_reciprocity(scheme; kind, ...) -> NamedTuple(scheme, kind, relL2, peak_fwd, peak_bwd, n)

Compare forward kernel(A->B) against backward kernel(B->A) on the negated field. `kind`:
  :diffusion_h  — U=V=0, horizontal Kh only (nz=1)  -> expect ~machine-zero (symmetric)
  :diffusion_v  — U=V=0, vertical Kz only            -> expect ~machine-zero (CN symmetric)
  :advection    — steady UNIFORM diagonal flow + Kh   -> EXACT (constant-velocity reversal == transpose)
  :advection_rot— steady solid-body ROTATION + Kh     -> ~truncation (spatially-varying flow: the
                  realistic band, since the real curvilinear/tidal flow is non-uniform)
"""
function bench_reciprocity(scheme::Symbol=:FFSL; kind::Symbol=:advection)
    if kind === :diffusion_h
        nx = ny = 32; nz = 1; dx = 100.0; A = (12, 16, 1); B = (20, 16, 1)
        ufun = (x, y) -> 0.0; vfun = (x, y) -> 0.0
        Kh = 3.0; Kz = 0.0; tmax = 20_000.0; dt = 500.0; store = 2_000.0
    elseif kind === :diffusion_v
        nx = ny = 6; nz = 12; dx = 100.0; A = (3, 3, 3); B = (3, 3, 9)
        ufun = (x, y) -> 0.0; vfun = (x, y) -> 0.0
        Kh = 0.0; Kz = 1e-3; tmax = 200.0; dt = 10.0; store = 20.0
    elseif kind === :advection_rot
        # solid-body rotation about the domain centre; A (west) advects a quarter-turn to B (south).
        nx = ny = 60; nz = 1; dx = 100.0; A = (15, 30, 1); B = (30, 15, 1)
        L = nx * dx; cx = L / 2; cy = L / 2; Om = 2π / 20_000.0
        ufun = (x, y) -> -Om * (y - cy); vfun = (x, y) -> Om * (x - cx)
        Kh = 1.0; Kz = 0.0; tmax = 5_000.0; dt = 30.0; store = 500.0
    else # :advection (uniform)
        nx = ny = 48; nz = 1; dx = 100.0; A = (14, 14, 1); B = (30, 24, 1)
        ufun = (x, y) -> 0.06; vfun = (x, y) -> 0.0375
        Kh = 1.0; Kz = 0.0; tmax = 26_667.0; dt = 500.0; store = 2_000.0
    end
    fwd = _spike_series(scheme, ufun, vfun, A, B; nx=nx, ny=ny, nz=nz, dx=dx, depth=10.0,
                        Kh=Kh, Kz=Kz, tmax=tmax, dt=dt, store=store)
    negu = (x, y) -> -ufun(x, y); negv = (x, y) -> -vfun(x, y)
    bwd = _spike_series(scheme, negu, negv, B, A; nx=nx, ny=ny, nz=nz, dx=dx, depth=10.0,
                        Kh=Kh, Kz=Kz, tmax=tmax, dt=dt, store=store)
    return (scheme=scheme, kind=kind, relL2=_rel_l2(fwd, bwd),
            peak_fwd=maximum(fwd), peak_bwd=maximum(bwd), n=min(length(fwd), length(bwd)))
end

# final physical tracer field after a spike run (optionally in reverse_time mode)
function _final_field(scheme::Symbol, ufun, vfun, src::NTuple{3,Int};
                      nx::Int, ny::Int, nz::Int, dx::Float64, depth::Float64,
                      Kh::Float64, Kz::Float64, tmax::Float64, dt::Float64,
                      reverse_time::Bool=false, origin::Float64=0.0)
    return mktempdir() do dir
        path = joinpath(dir, "vel.nc")
        write_velocity_nc(path; nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, depth=depth,
                          ufun=ufun, vfun=vfun, tmax=tmax * 10)
        local out
        quiet() do
            grid, ds, hydro, state = build_grid_state(path)
            ng = grid.ng
            fill!(state.tracers[:C], 0.0)
            state.tracers[:C][src[1]+ng, src[2]+ng, src[3]] = 1.0
            final = run_simulation(grid, state, PointSource[], 0.0, tmax, dt;
                                   ds=ds, hydro_data=hydro, advection_scheme=scheme,
                                   use_adaptive_dt=false, Kh=Kh, Kz=Kz,
                                   diagnose_vertical_velocity=false, write_full_state=false,
                                   reverse_time=reverse_time, reverse_time_origin=origin)
            close(ds)
            out = Array(phys_view(final.tracers[:C], grid))
        end
        return out
    end
end

"""
    bench_reverse_time_equivalence(scheme) -> NamedTuple(maxabs, rel, peak)

Plumbing check for the `reverse_time` solver flag: on a STEADY spatially-varying (rotation) field,
a reverse_time run on the ORIGINAL field must be identical (to machine precision) to a forward run on
the pre-NEGATED field — both integrate the same effective velocity each step. This isolates the
velocity-negation + hydro-time-mapping plumbing from the (already validated) reciprocity principle.
"""
function bench_reverse_time_equivalence(scheme::Symbol=:FFSL)
    nx = ny = 60; nz = 1; dx = 100.0; S = (30, 15, 1)
    L = nx * dx; cx = L / 2; cy = L / 2; Om = 2π / 20_000.0
    ufun = (x, y) -> -Om * (y - cy); vfun = (x, y) -> Om * (x - cx)
    Kh = 1.0; Kz = 0.0; tmax = 5_000.0; dt = 30.0
    negu = (x, y) -> -ufun(x, y); negv = (x, y) -> -vfun(x, y)
    F_ref = _final_field(scheme, negu, negv, S; nx=nx, ny=ny, nz=nz, dx=dx, depth=10.0,
                         Kh=Kh, Kz=Kz, tmax=tmax, dt=dt)                                   # forward on negated field
    F_rev = _final_field(scheme, ufun, vfun, S; nx=nx, ny=ny, nz=nz, dx=dx, depth=10.0,
                         Kh=Kh, Kz=Kz, tmax=tmax, dt=dt, reverse_time=true, origin=tmax)   # reverse_time on original
    d = maximum(abs.(F_ref .- F_rev)); scale = maximum(abs.(F_ref))
    return (scheme=scheme, maxabs=d, rel=scale > 0 ? d / scale : d, peak=scale)
end
