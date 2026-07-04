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
using HydrodynamicTransport.ProjectionModule: build_projector
using HydrodynamicTransport.BreathingTransportModule: _zsweep_vffsl!
using NCDatasets
using Random
using LinearAlgebra

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

# Build the LINEAR breathing sweep operator as an explicit nphys×nphys matrix by applying the production
# kernel `_ffsl_line_breathing!` (linear mode) to each physical-cell unit basis vector. Departure volumes
# `vd`, swept volumes `Ss` (closed line: 0 at the physical-block boundary faces). Top-level (NOT a nested
# closure — a nested closure mis-inferred the scratch and silently corrupted the second call's result).
function _breathing_sweep_matrix(vd::Vector{Float64}, Ss::Vector{Float64}, m::Int, ng::Int, nphys::Int)
    _line = HydrodynamicTransport.HorizontalTransportModule._ffsl_line_breathing!
    M = zeros(nphys, nphys)
    for j in 1:nphys
        crow = zeros(m); crow[j+ng] = 1.0
        varr_s = zeros(m); cL = zeros(m); cR = zeros(m); Flo = zeros(m); Fhi = zeros(m)
        Ctd = zeros(m); Rp = zeros(m); Rm = zeros(m); Srow = copy(Ss)
        _line(crow, crow, vd, varr_s, Srow, cL, cR, Flo, Fhi, Ctd, Rp, Rm, m, ng, nphys;
              camb=0.0, linear=true)
        for i in 1:nphys; M[i, j] = crow[i+ng]; end
    end
    return M
end

# bench_breathing_adjoint_kernel(; nphys, ng, seed, amp) -> (resid, rel, courant, nphys)
#
# MACHINE-PRECISION adjoint of the LINEAR (`linear=true`) breathing donor-cell sweep, tested directly on
# the production kernel `_ffsl_line_breathing!` with NO external data. Builds the sweep operator M on a
# closed 1-D breathing line (random departure volumes vdep>0, a smooth swept-volume profile S that
# vanishes a buffer of cells inside the physical block so the donor walk never touches a ghost cell) and
# its reverse-time form Mt (negate S, use the arrival volumes varr as the reverse departure), each as an
# explicit nphys x nphys matrix. Returns the volume-weighted transpose residual
# max|diag(varr)M - (diag(vdep)Mt)ᵀ|, which is ~machine-zero — the exact discrete-adjoint property
# (math-vetted by 3 independent agents: exact at any Courant while every cell volume stays positive).
# `amp` scales the peak Courant (amp<1 => single-cell; amp>1 => multi-cell).
function bench_breathing_adjoint_kernel(; nphys::Int=48, ng::Int=2, seed::Int=1234, amp::Float64=0.6)
    m = nphys + 2ng
    rng = MersenneTwister(seed)
    vdep = 0.5 .+ 1.5 .* rand(rng, m)                    # departure volumes in [0.5, 2.0]
    vmin = minimum(vdep)
    # Smooth swept-volume profile S[f] = amp*vmin*sin(2π(f-lo)/(hi-lo)) on the interior faces lo..hi, 0
    # elsewhere: vanishes at both ends (a closed line, matching the abstract S[0]=S[m]=0 model), has a
    # small face-to-face slope (so all arrival volumes stay positive), yet reaches Courant≈amp at its
    # peak and changes sign (both walk directions). `buffer` keeps the donor walk clear of the ghosts.
    buffer = max(3, ceil(Int, amp) + 2)
    lo = ng + buffer; hi = m - ng - buffer
    S = zeros(m)
    for f in lo:hi
        S[f] = amp * vmin * sinpi(2 * (f - lo) / (hi - lo))
    end
    varr = copy(vdep)
    for g in 2:m-1; varr[g] = vdep[g] - (S[g] - S[g-1]); end
    @assert all(>(0.0), varr) "test setup produced a non-positive arrival volume (widen the line)"
    peak_courant = maximum(abs(S[f]) / min(vdep[f], vdep[f+1]) for f in 1:m-1)

    M  = _breathing_sweep_matrix(vdep, S, m, ng, nphys)   # forward operator
    Mt = _breathing_sweep_matrix(varr, -S, m, ng, nphys)  # reverse-time: departure = forward arrival, swept = -S
    a = varr[(ng+1):(ng+nphys)]; d = vdep[(ng+1):(ng+nphys)]
    LHS = Diagonal(a) * M
    RHS = permutedims(Diagonal(d) * Mt)
    resid = maximum(abs.(LHS .- RHS))
    scale = maximum(abs.(LHS))
    return (resid=resid, rel=(scale > 0 ? resid / scale : resid), courant=peak_courant, nphys=nphys)
end

# fresh per-call scratch tuple for `_zsweep_vffsl!` (matches the layout built in breathing_transport!)
_vffsl_scratch(mmax, nz) = (crow=zeros(mmax), Srow=zeros(mmax), varr=zeros(mmax), cL=zeros(mmax), cR=zeros(mmax),
    Flo=zeros(mmax), Fhi=zeros(mmax), Ctd=zeros(mmax), Rp=zeros(mmax), Rm=zeros(mmax), dl=zeros(nz), dd=zeros(nz),
    du=zeros(nz), rhs=zeros(nz), cprime=zeros(nz), om=zeros(nz+1), Vdc=zeros(nz), Va=zeros(nz), dz=zeros(nz),
    vcol=zeros(max(mmax, nz+2)))

# Build the vertical z-step column operator (nz×nz) at cell (ic,jc): set ω = omsign·om on every active
# column, seed a unit basis in each layer, apply `_zsweep_vffsl!`, read the (ic,jc) column. TOP-LEVEL (a
# nested closure silently corrupts the 2nd call — the Julia boxing gotcha this file already hit once).
function _vffsl_zop(proj, grid, om, Vd3vol, omsign, dt, Kz, ic, jc, linear)
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
    for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        for k in 1:nz+1; proj.omega[i, j, k] = omsign * om[k]; end
    end
    Vd3 = zeros(nx+2ng, ny+2ng, nz)
    for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        for k in 1:nz; Vd3[i+ng, j+ng, k] = Vd3vol[k]; end
    end
    M = zeros(nz, nz); s = _vffsl_scratch(nz+2, nz)
    for kk in 1:nz
        C = zeros(nx+2ng, ny+2ng, nz); C[ic+ng, jc+ng, kk] = 1.0
        _zsweep_vffsl!(C, proj, Vd3, dt, Kz, s, ng, nx, ny, nz, linear)
        for k in 1:nz; M[k, kk] = C[ic+ng, jc+ng, k]; end
    end
    return M
end

# bench_vertical_ffsl_adjoint(; nz, seed) -> (adj, c1_lin, c1_ppm)
#
# Validate the opt-in vertical FFSL z-step (`breathing_vffsl`) with NO external data: on a deep uniform
# column (interior active, ω=0 at seabed/surface), (1) the LINEAR z-step operator is EXACTLY self-adjoint
# under reverse time — `diag(Va)·M = (diag(Vd)·M_rev)ᵀ` with M_rev = negate ω, depart from Va (math-vetted,
# 3 agents), and (2) C≡1 is preserved bit-exact in BOTH the linear (donor-cell) and PPM+FCT flux modes.
function bench_vertical_ffsl_adjoint(; nz::Int=6, seed::Int=5)
    nx = ny = 5; ng = 2; dx = 100.0; Lz = 30.0
    nxt, nyt = nx + 2ng, ny + 2ng
    pm = fill(1/dx, nxt, nyt); pn = fill(1/dx, nxt, nyt); h = fill(Lz, nxt, nyt); z = zeros(nxt, nyt)
    z_w = collect(range(-Lz, 0.0, length=nz+1)); vol = zeros(nxt, nyt, nz); fax = zeros(nxt+1, nyt, nz); fay = zeros(nxt, nyt+1, nz)
    for k in 1:nz; dz = abs(z_w[k+1]-z_w[k]); vol[:, :, k] .= dx*dx*dz; fax[:, :, k] .= dx*dz; fay[:, :, k] .= dx*dz; end
    grid = CurvilinearGrid(ng, nx, ny, nz, z, z, z, z, z, z, z_w, pm, pn, z, h,
        trues(nxt, nyt), trues(nx+1+2ng, nyt), trues(nxt, ny+1+2ng), fax, fay, vol)
    proj = build_projector(grid; h_open=20.0)      # deep edges = Dirichlet mouth ⇒ interior active
    ic, jc = 3, 3
    a = proj.dxo[ic, jc] * proj.dyo[ic, jc]; dt = 100.0; Kz = 1e-3
    rng = MersenneTwister(seed)
    om = zeros(nz+1); for k in 2:nz; om[k] = (2rand(rng)-1)*3.0e2; end
    Vd = [a*(0.8+0.6rand(rng))*(30.0/nz) for _ in 1:nz]
    Va = [Vd[k] - dt*(om[k+1]-om[k]) for k in 1:nz]
    M  = _vffsl_zop(proj, grid, om, Vd, +1.0, dt, Kz, ic, jc, true)
    Mr = _vffsl_zop(proj, grid, om, Va, -1.0, dt, Kz, ic, jc, true)
    adj = maximum(abs.(Diagonal(Va) * M .- permutedims(Diagonal(Vd) * Mr)))
    c1 = Float64[]
    for lin in (true, false)
        for j in 1:ny, i in 1:nx
            proj.active[i, j] || continue
            for k in 1:nz+1; proj.omega[i, j, k] = om[k]; end
        end
        Vd3 = zeros(nx+2ng, ny+2ng, nz)
        for j in 1:ny, i in 1:nx; proj.active[i, j] || continue; for k in 1:nz; Vd3[i+ng, j+ng, k] = Vd[k]; end; end
        C = ones(nx+2ng, ny+2ng, nz)
        _zsweep_vffsl!(C, proj, Vd3, dt, Kz, _vffsl_scratch(nz+2, nz), ng, nx, ny, nz, lin)
        m1 = 0.0
        for j in 1:ny, i in 1:nx; proj.active[i, j] || continue; for k in 1:nz; m1 = max(m1, abs(C[i+ng, j+ng, k]-1.0)); end; end
        push!(c1, m1)
    end
    return (adj=adj, va_positive=all(>(0.0), Va), c1_lin=c1[1], c1_ppm=c1[2])
end
