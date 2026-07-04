# src/ProjectionModule.jl
#
# Depth-weighted barotropic Helmholtz/Poisson projection for the OPT-IN breathing-sigma
# continuity correction. This is the production port of the validated proof-of-concept
# `claude/breathing_sigma_poc/stage1_projection.jl` + the projection/omega parts of
# `stage2_c1_breathing.jl` (see the plan in
# `~/.claude/plans/inhydrodynamictransport-i-gt-omments-iridescent-teapot.md`).
#
# WHY. The offline solver is rigid-lid: `grid.volume`/`face_area` are frozen at H0 and never
# breathe with the free surface, so the archived 30-min velocity snapshots do not satisfy discrete
# continuity vs `∂η/∂t`. A conserved tracer then develops a ~0.5 dex tide-locked magnitude error at
# the intertidal receptor and there is no tidal dilution. This module uses the RELIABLE `∂η/∂t`
# (from `XE`/zeta) to minimally, curl-freely correct the transport so its discrete divergence equals
# `−Area·∂η/∂t` EXACTLY, per layer summed. The correction is a barotropic potential-flow field
# `u' = D∇φ` distributed depth-uniformly (∝ Δσ_k), which preserves the baroclinic shear.
#
# The scheme (3-solver-unanimous, PoC-validated):
#   solve   Σ_{n~i} w_f (φ_i − φ_n) = T_i − Draw_i ,   w_f = L_f·D_f/Δ_f    (∇·(D∇φ) form)
#   with    D_f = min(D_i, D_n)  (face depth, conservative, zero over land),
#           T_i = −A_i·∂η/∂t      (the target divergence — the tidal prism), and
#           Draw_i = the raw (uncorrected) depth-integrated horizontal divergence.
# Boundary conditions: DEEP-MOUTH (`h > h_open`) wet-region-edge cells are Dirichlet φ=0 — this is
# MANDATORY because ΣT ≠ 0 (the prism has to exit somewhere); land/isolated faces are Neumann
# (dropped); only the mouth-connected wet component (flood-filled from the Dirichlet seeds) is solved.
#
# Corrected per-layer face transports and the GCL vertical volume flux ω (bottom-up, closing at the
# surface because Σ_k div_h U* = T) are produced for the breathing-FFSL kernels. All transports are
# VOLUME fluxes [m³/s].

module ProjectionModule

export BreathingProjector, build_projector, project!, continuity_residual, correction_ratio_quantiles,
       padded_depth!, write_omega_velocity!

using ..HydrodynamicTransport.ModelStructs
using SparseArrays, LinearAlgebra

"""
    BreathingProjector

Persistent per-simulation workspace for the barotropic projection. The STATIC topology (wet mask,
deep-mouth Dirichlet set, mouth-connected reachable component, interior-unknown numbering, and the
horizontal metrics) is built once by [`build_projector`](@ref). Each 30-min hydro read,
[`project!`](@ref) recomputes the depth field `D̃` from the bracketing surfaces, (re)assembles and
factorizes the symmetric SPD Poisson operator, solves for φ, and fills the corrected per-layer face
transports `Ux`,`Uy` and the GCL vertical flux `omega`, plus the floored breathing volumes `Vn`,`Vnp1`.

All index arrays are PHYSICAL (1:nx, 1:ny); grid arrays are addressed with the `+ng` ghost offset.
Transports are volume fluxes [m³/s]; `Ux[i,j,k]` is the WEST face of cell (i,j), `Uy[i,j,k]` the SOUTH
face, `omega[i,j,k]` the bottom face of layer k (k=1 seabed, k=nz+1 surface).
"""
mutable struct BreathingProjector
    nx::Int; ny::Int; nz::Int; ng::Int
    h_open::Float64          # deep-mouth Dirichlet threshold [m]
    wet_min::Float64         # bathymetry wet threshold [m]
    D_min::Float64           # floor on the total water depth D̃ [m]
    # --- wet/dry parking (opt-in) ---
    parking::Bool            # dynamic wet/dry parking on? (false = static topology, bit-identical)
    D_park::Float64          # park a wet cell whose min(Hn,Hnp) < D_park (> D_min)
    parked::BitMatrix        # (nx,ny) cells frozen this read (shallow OR mouth-disconnected). walls.
    # --- topology (STATIC when parking=false; the wet/isopen masks + metrics are always static, but
    #     reach/active/id/N are RECOMPUTED per read when parking=true — the parked set is dynamic) ---
    wet::BitMatrix           # (nx,ny) mask_rho ∧ h>wet_min
    isopen::BitMatrix        # deep-mouth Dirichlet cells (φ=0)
    reach::BitMatrix         # mouth-connected component (of `wet`, or `wet∧¬parked` when parking)
    active::BitMatrix        # interior unknowns (reach ∧ ¬isopen ∧ ¬parked)
    id::Matrix{Int}          # (nx,ny) → unknown index (0 if not an unknown)
    N::Int                   # number of unknowns
    # --- horizontal metrics (physical cells) ---
    dxo::Matrix{Float64}     # 1/pm  [m]
    dyo::Matrix{Float64}     # 1/pn  [m]
    dsig::Vector{Float64}    # Δσ_k (Σ = 1 on a sigma grid)  (nz,)
    # --- per-read work ---
    D::Matrix{Float64}       # floored total depth at mid-interval [m]
    Draw::Matrix{Float64}    # raw depth-integrated horizontal divergence [m³/s]
    T::Matrix{Float64}       # target divergence −A·∂η/∂t [m³/s]
    Hn::Matrix{Float64}      # floored total depth at t_n [m]
    Hnp::Matrix{Float64}     # floored total depth at t_{n+1} [m]
    phi::Matrix{Float64}     # projection potential
    Ux::Array{Float64,3}     # corrected west-face transport [m³/s]  (nx+1,ny,nz)
    Uy::Array{Float64,3}     # corrected south-face transport [m³/s] (nx,ny+1,nz)
    omega::Array{Float64,3}  # vertical volume flux, bottom-up [m³/s] (nx,ny,nz+1)
    depth_pad::Matrix{Float64}  # padded floored depth for rebuild_metrics! (nx_tot,ny_tot)
    last_idx::Int            # bracket index of the last projection (cadence guard; -1 = none yet)
    t_read_start::Float64    # time [s] at the start of the current hydro read (for the sub-step fraction)
    t_read_end::Float64      # time [s] at the end of the current hydro read
    # --- linear solve scratch (COO reassembled per read; pattern is static) ---
    b::Vector{Float64}
end

# --- accessors into the padded grid arrays ---
@inline _inb(p::BreathingProjector, i, j) = 1 <= i <= p.nx && 1 <= j <= p.ny
@inline _wet(p::BreathingProjector, i, j) = _inb(p, i, j) && @inbounds(p.wet[i, j])
@inline _isopen(p::BreathingProjector, i, j) = _inb(p, i, j) && @inbounds(p.isopen[i, j])
@inline _active(p::BreathingProjector, i, j) = _inb(p, i, j) && @inbounds(p.active[i, j])
@inline _parked(p::BreathingProjector, i, j) = _inb(p, i, j) && @inbounds(p.parked[i, j])
# transportable = wet and NOT parked; faces are open to transport only between two transportable cells.
@inline _transp(p::BreathingProjector, i, j) = _inb(p, i, j) && @inbounds(p.wet[i, j] && !p.parked[i, j])

# Depth of cell (i,j) — floored total water depth (0 over land). Read from proj.D (filled per read).
@inline _Dc(p::BreathingProjector, i, j) = _wet(p, i, j) ? @inbounds(p.D[i, j]) : 0.0

# Face weight w_f = L_f·D_f/Δ_f of the ∇·(D∇φ) operator on the face between (i,j) and (ni,nj).
@inline function _facew(p::BreathingProjector, i, j, ni, nj)
    Df = min(_Dc(p, i, j), _Dc(p, ni, nj))
    if ni != i    # x-face: transverse length = dyo, centre spacing = dxo
        Lf = 0.5 * (p.dyo[i, j] + p.dyo[ni, nj]); dc = 0.5 * (p.dxo[i, j] + p.dxo[ni, nj])
    else          # y-face: transverse length = dxo, centre spacing = dyo
        Lf = 0.5 * (p.dxo[i, j] + p.dxo[ni, nj]); dc = 0.5 * (p.dyo[i, j] + p.dyo[ni, nj])
    end
    return Lf * Df / dc
end

"""
    build_projector(grid; h_open=30.0, wet_min=1.0, D_min=0.1) -> BreathingProjector

Build the static projection topology from `grid`: the wet mask, the deep-mouth Dirichlet seeds
(`h > h_open`), the mouth-connected wet component (flood-fill), the interior-unknown numbering, and
the horizontal metrics. This is done ONCE per simulation; the geometry-dependent linear operator is
(re)assembled per hydro read inside [`project!`](@ref) because its coefficient `D̃` breathes with η.
"""
function build_projector(grid::CurvilinearGrid; h_open::Float64=30.0, wet_min::Float64=1.0,
                         D_min::Float64=0.1, parking::Bool=false, D_park::Float64=0.5)
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz

    wet = falses(nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        wet[i, j] = grid.mask_rho[i+ng, j+ng] && grid.h[i+ng, j+ng] > wet_min
    end
    # deep-mouth open boundary: a wet cell that touches a non-wet neighbour AND is deep (h > h_open).
    isopen = falses(nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        wet[i, j] || continue
        touches_nonwet = !( (i > 1  && wet[i-1, j]) && (i < nx && wet[i+1, j]) &&
                            (j > 1  && wet[i, j-1]) && (j < ny && wet[i, j+1]) )
        isopen[i, j] = touches_nonwet && grid.h[i+ng, j+ng] > h_open
    end

    dxo = Matrix{Float64}(undef, nx, ny); dyo = Matrix{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        dxo[i, j] = 1.0 / grid.pm[i+ng, j+ng]; dyo[i, j] = 1.0 / grid.pn[i+ng, j+ng]
    end
    # Δσ_k from the sigma interfaces (Σ = 1 on a sigma grid, spanning [-1,0]).
    dsig = Float64[grid.z_w[k+1] - grid.z_w[k] for k in 1:nz]

    # N_max = all-wet interior count sizes the RHS scratch; the per-read active count N ≤ N_max.
    Nmax = count(wet) - count(isopen)
    nx_tot, ny_tot = size(grid.pm)
    proj = BreathingProjector(nx, ny, nz, ng, h_open, wet_min, D_min,
        parking, D_park, falses(nx, ny),
        wet, isopen, falses(nx, ny), falses(nx, ny), zeros(Int, nx, ny), 0,
        dxo, dyo, dsig,
        zeros(nx, ny), zeros(nx, ny), zeros(nx, ny), zeros(nx, ny), zeros(nx, ny), zeros(nx, ny),
        zeros(nx+1, ny, nz), zeros(nx, ny+1, nz), zeros(nx, ny, nz+1),
        zeros(nx_tot, ny_tot), -1, 0.0, 0.0,
        zeros(max(Nmax, 0)))
    # Static topology with an empty parked set (parking=false keeps this every read — bit-identical).
    _flood_and_number!(proj)
    return proj
end

# (Re)compute the mouth-connected component + interior-unknown numbering from the CURRENT transportable
# set `wet ∧ ¬parked`. `isopen` (deep mouth) never parks and seeds the flood-fill. Any transportable cell
# the single global flood-fill does NOT reach is mouth-disconnected — it is folded into `parked` (a wall):
# a breathing Neumann island has Σ(T−Draw) = −ΔV_pool/Δt ≠ 0 (unsolvable) and adds a constant null-vector
# (singular), so parking it is required for a well-posed SPD system (agent-vetted). Removing non-reachable
# cells cannot disconnect reachable ones (reachability is monotone) ⇒ ONE pass is a fixed point, no
# iteration. Fills proj.reach, proj.active, proj.id, proj.N; extends proj.parked with the disconnected set.
function _flood_and_number!(proj::BreathingProjector)
    nx, ny = proj.nx, proj.ny
    wet, isopen, parked = proj.wet, proj.isopen, proj.parked
    reach = proj.reach; fill!(reach, false)
    stack = Tuple{Int,Int}[]
    @inbounds for j in 1:ny, i in 1:nx
        if isopen[i, j]; reach[i, j] = true; push!(stack, (i, j)); end
    end
    while !isempty(stack)
        (i, j) = pop!(stack)
        for (ni, nj) in ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
            if 1 <= ni <= nx && 1 <= nj <= ny && wet[ni, nj] && !parked[ni, nj] && !reach[ni, nj]
                reach[ni, nj] = true; push!(stack, (ni, nj))
            end
        end
    end
    active = proj.active; id = proj.id
    fill!(active, false); fill!(id, 0); N = 0
    @inbounds for j in 1:ny, i in 1:nx
        active[i, j] = reach[i, j] && !isopen[i, j] && !parked[i, j]
        if active[i, j]; N += 1; id[i, j] = N; end
        # PARKING ONLY: fold mouth-disconnected transportable cells into the parked (wall) set so their
        # faces get zeroed (a breathing Neumann island is otherwise unsolvable). Not done when parking is
        # off — there disconnected wet cells keep the legacy raw-transport treatment (bit-identical).
        if proj.parking && wet[i, j] && !isopen[i, j] && !reach[i, j]; parked[i, j] = true; end
    end
    proj.N = N
    return proj
end

"""
    project!(proj, grid, u, v, eta_n, eta_np1, dt) -> proj

Run the barotropic projection for one hydro interval. `u`,`v` are the (padded, ghost-offset) layer
velocities [m/s] already loaded into `state.u`/`state.v` (mid-interval); `eta_n`,`eta_np1` are the
PHYSICAL (nx,ny) free-surface snapshots at the bracketing reads [m]; `dt = ΔT` is the read interval [s].

Fills, in `proj`: `D` (mid-interval floored depth), `Hn`,`Hnp` (floored end depths), `Draw`,`T`, `phi`,
the corrected per-layer face transports `Ux`,`Uy`, and the bottom-up GCL vertical flux `omega`. After
this call `Σ_k (Ux[i+1]−Ux[i]+Uy[j+1]−Uy[j]) = T` to solver tolerance on every active cell, and the
per-cell GCL `V^{n+1}−V^n = −Δt·div(U*,ω)` holds by construction (see [`continuity_residual`](@ref)).
"""
function project!(proj::BreathingProjector, grid::CurvilinearGrid,
                  u::AbstractArray{Float64,3}, v::AbstractArray{Float64,3},
                  eta_n::AbstractMatrix, eta_np1::AbstractMatrix, dt::Float64)
    nx, ny, nz, ng = proj.nx, proj.ny, proj.nz, proj.ng
    D, Draw, T, Hn, Hnp = proj.D, proj.Draw, proj.T, proj.Hn, proj.Hnp
    dxo, dyo, dsig = proj.dxo, proj.dyo, proj.dsig
    Dmin = proj.D_min

    # mid-interval and end-of-interval floored total depths
    @inbounds for j in 1:ny, i in 1:nx
        if proj.wet[i, j]
            hb = grid.h[i+ng, j+ng]
            emid = 0.5 * (eta_n[i, j] + eta_np1[i, j])
            D[i, j]  = max(hb + emid, Dmin)
            Hn[i, j] = max(hb + eta_n[i, j], Dmin)
            Hnp[i, j] = max(hb + eta_np1[i, j], Dmin)
        else
            D[i, j] = 0.0; Hn[i, j] = 0.0; Hnp[i, j] = 0.0
        end
    end

    # WET/DRY PARKING (opt-in): freeze cells that go shallow during this read + any cell the parking
    # disconnects from the mouth. The mask is a pure function of the (floored) depths min(Hn,Hnp) — which
    # is SWAP-INVARIANT under the reverse-time Hn↔Hnp swap, so forward and reverse project the identical
    # mask automatically (agent-vetted). Parked cells become no-flux walls: excluded from the Poisson
    # unknowns, their raw face transports zeroed (so Draw is masked consistently — the load-bearing bug to
    # avoid), and frozen (V0=0, C held) in the cascade. min(Hn,Hnp) < D_park (> D_min) also guarantees
    # every active cell has strictly positive face depths ⇒ SPD. Recompute the reach/active/id topology.
    if proj.parking
        parked = proj.parked; fill!(parked, false)
        @inbounds for j in 1:ny, i in 1:nx
            parked[i, j] = proj.wet[i, j] && min(Hn[i, j], Hnp[i, j]) < proj.D_park
        end
        _flood_and_number!(proj)   # re-flood-fill on wet∧¬parked; folds mouth-disconnected cells into parked
    end

    # RAW per-layer face transports (uncorrected). Single-valued per face: the west face of cell
    # (i,j) uses the face-averaged transverse length and the min face depth. Building the raw transport
    # FIRST — then deriving Draw as its exact discrete divergence — is what makes Draw, the Poisson
    # stencil, and the corrected transport mutually consistent, so div(U*) = T to solver tolerance and
    # the GCL ω closes at the surface to machine zero. (The PoC computed Draw with the cell-centred
    # metric, which is only self-consistent on a uniform grid; it never tested the layer divergence.)
    # Faces touching a PARKED cell carry zero raw transport (Neumann wall) via `_transp` — this masks
    # Draw consistently with the walled Poisson stencil, so div(U*)=T and C≡1 still hold on active cells.
    Ux, Uy = proj.Ux, proj.Uy
    fill!(Ux, 0.0); fill!(Uy, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        if _transp(proj, i-1, j) && _transp(proj, i, j)
            ig, jg = i + ng, j + ng
            dyf = 0.5 * (dyo[i-1, j] + dyo[i, j]); Df = min(_Dc(proj, i-1, j), _Dc(proj, i, j))
            for k in 1:nz; Ux[i, j, k] = u[ig, jg, k] * dyf * dsig[k] * Df; end
        end
        if _transp(proj, i, j-1) && _transp(proj, i, j)
            ig, jg = i + ng, j + ng
            dxf = 0.5 * (dxo[i, j-1] + dxo[i, j]); Df = min(_Dc(proj, i, j-1), _Dc(proj, i, j))
            for k in 1:nz; Uy[i, j, k] = v[ig, jg, k] * dxf * dsig[k] * Df; end
        end
    end

    # raw depth-integrated horizontal divergence Draw (from the raw transports) and target T = −A·∂η/∂t
    @inbounds for j in 1:ny, i in 1:nx
        if !proj.wet[i, j]; Draw[i, j] = 0.0; T[i, j] = 0.0; continue; end
        s = 0.0
        for k in 1:nz
            s += Ux[i+1, j, k] - Ux[i, j, k] + Uy[i, j+1, k] - Uy[i, j, k]
        end
        Draw[i, j] = s
        # Target = −A·∂D̃/∂t using the FLOORED depth tendency (Hnp−Hn), NOT the raw ∂η/∂t. This is
        # the single consistent floor (plan §5): at drying cells where D̃ is clipped to D_min, the
        # tendency (Hnp−Hn) matches the breathing volume increment dVk exactly, so the GCL ω closes to
        # machine zero everywhere (deep cells: Hnp−Hn = η_np1−η_n, so this is a no-op there).
        T[i, j] = -dxo[i, j] * dyo[i, j] * (Hnp[i, j] - Hn[i, j]) / dt
    end

    # assemble the symmetric SPD Poisson operator on the interior unknowns and solve
    N = proj.N
    II = Int[]; JJ = Int[]; VV = Float64[]
    sizehint!(II, 5N); sizehint!(JJ, 5N); sizehint!(VV, 5N)
    b = proj.b; fill!(b, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        _active(proj, i, j) || continue
        p = proj.id[i, j]; diag = 0.0
        for (ni, nj) in ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
            if _active(proj, ni, nj)
                w = _facew(proj, i, j, ni, nj); diag += w
                push!(II, p); push!(JJ, proj.id[ni, nj]); push!(VV, -w)
            elseif _isopen(proj, ni, nj)            # Dirichlet φ=0 (mouth): diagonal only
                diag += _facew(proj, i, j, ni, nj)
            end                                     # else land/isolated: Neumann (dropped)
        end
        push!(II, p); push!(JJ, p); push!(VV, diag)
        b[p] = T[i, j] - Draw[i, j]
    end
    phi = proj.phi; fill!(phi, 0.0)
    if N > 0
        A = sparse(II, JJ, VV, N, N)
        phiv = A \ b[1:N]    # b is sized to N_max (all-wet); the active count N ≤ N_max varies per read
        @inbounds for j in 1:ny, i in 1:nx
            _active(proj, i, j) && (phi[i, j] = phiv[proj.id[i, j]])
        end
    end

    # add the barotropic potential-flow correction u' = w_f(φ_L−φ_R) to the raw transports,
    # distributed depth-uniformly (∝ Δσ_k, Σ_k = 1) so the baroclinic shear is preserved. The
    # correction lives only on faces internal to the mouth-connected component (both ends active
    # or Dirichlet-open); land/isolated faces are Neumann (no correction), matching the stencil.
    @inbounds for j in 1:ny, i in 1:nx
        # west face of cell (i,j): between (i-1,j) and (i,j)
        if _wet(proj, i-1, j) && _wet(proj, i, j) &&
           (_active(proj, i, j) || _isopen(proj, i, j)) &&
           (_active(proj, i-1, j) || _isopen(proj, i-1, j))
            q = _facew(proj, i, j, i-1, j) * (phi[i-1, j] - phi[i, j])
            for k in 1:nz; Ux[i, j, k] += q * dsig[k]; end
        end
        # south face of cell (i,j): between (i,j-1) and (i,j)
        if _wet(proj, i, j-1) && _wet(proj, i, j) &&
           (_active(proj, i, j) || _isopen(proj, i, j)) &&
           (_active(proj, i, j-1) || _isopen(proj, i, j-1))
            q = _facew(proj, i, j, i, j-1) * (phi[i, j-1] - phi[i, j])
            for k in 1:nz; Uy[i, j, k] += q * dsig[k]; end
        end
    end

    # GCL vertical volume flux ω, bottom-up per column, from the SAME corrected transports.
    # ω_{k+½} = ω_{k−½} − (Δσ_k·A·∂D̃/∂t + div_h U*_k),  ω_bed = 0.  Closes at the surface
    # (ω_top ≈ 0) because Σ_k div_h U*_k = T = −A·∂η/∂t.
    omega = proj.omega; fill!(omega, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        (_active(proj, i, j) || _isopen(proj, i, j)) || continue
        area = dxo[i, j] * dyo[i, j]
        dHdt = (Hnp[i, j] - Hn[i, j]) / dt
        for k in 1:nz
            hdiv = Ux[i+1, j, k] - Ux[i, j, k] + Uy[i, j+1, k] - Uy[i, j, k]
            dVk = area * dsig[k] * dHdt
            omega[i, j, k+1] = omega[i, j, k] - (dVk + hdiv)
        end
    end
    return proj
end

"""
    continuity_residual(proj) -> (max, mean)

Max/mean of `|div_h(U*) − T| / scale` over the active cells (scale = max(|T|,|Draw|,1e-30)). Diagnostic
for the PoC oracle: after [`project!`](@ref) this closes to ~machine zero (≈1e-10 on the real grid).
"""
function continuity_residual(proj::BreathingProjector)
    nx, ny, nz = proj.nx, proj.ny, proj.nz
    Ux, Uy, T, Draw = proj.Ux, proj.Uy, proj.T, proj.Draw
    maxres = 0.0; sumres = 0.0; n = 0
    @inbounds for j in 1:ny, i in 1:nx
        _active(proj, i, j) || continue
        div = 0.0
        for k in 1:nz
            div += Ux[i+1, j, k] - Ux[i, j, k] + Uy[i, j+1, k] - Uy[i, j, k]
        end
        sc = max(abs(T[i, j]), abs(Draw[i, j]), 1e-30)
        r = abs(div - T[i, j]) / sc
        maxres = max(maxres, r); sumres += r; n += 1
    end
    return (max=maxres, mean=sumres / max(n, 1))
end

"""
    correction_ratio_quantiles(proj, u) -> sorted Vector

Per-interior-x-face `|u'|/|u|` (barotropic correction magnitude relative to the depth-mean raw
velocity). The PoC oracle reports median ≈ 0.21 (a gentle, bounded correction).
"""
function correction_ratio_quantiles(proj::BreathingProjector, u::AbstractArray{Float64,3})
    nx, ny, nz, ng = proj.nx, proj.ny, proj.nz, proj.ng
    dxo, dyo, dsig, phi = proj.dxo, proj.dyo, proj.dsig, proj.phi
    rels = Float64[]
    @inbounds for j in 2:ny-1, i in 2:nx-1
        (_wet(proj, i, j) && _wet(proj, i-1, j) && proj.reach[i, j] && proj.reach[i-1, j]) || continue
        ig, jg = i + ng, j + ng
        dc = 0.5 * (dxo[i-1, j] + dxo[i, j])
        up = (phi[i-1, j] - phi[i, j]) / dc
        ubar = 0.0
        for k in 1:nz; ubar += u[ig, jg, k] * dsig[k]; end
        push!(rels, abs(up) / max(abs(ubar), 0.05))
    end
    sort!(rels)
    return rels
end

"""
    padded_depth!(proj, grid) -> proj.depth_pad

Fill the padded (`nx_tot × ny_tot`) breathing depth field from `proj.D` (the mid-interval floored
depth; 0 over land) and extrapolate into the ghost ring with the same edge-copy convention as
`initialize_curvilinear_grid`. Feed the result to [`rebuild_metrics!`](@ref). Call after [`project!`](@ref).
"""
function padded_depth!(proj::BreathingProjector, grid::CurvilinearGrid)
    ng, nx, ny = proj.ng, proj.nx, proj.ny
    dp = proj.depth_pad; fill!(dp, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        dp[i+ng, j+ng] = proj.wet[i, j] ? proj.D[i, j] : 0.0
    end
    nx_tot, ny_tot = size(dp)
    @inbounds for j in ng+1:ny+ng, g in 1:ng
        dp[g, j] = dp[ng+1, j]; dp[nx+ng+g, j] = dp[nx+ng, j]
    end
    @inbounds for i in 1:nx_tot, g in 1:ng
        dp[i, g] = dp[i, ng+1]; dp[i, ny+ng+g] = dp[i, ny+ng]
    end
    return dp
end

"""
    write_omega_velocity!(proj, grid, state) -> state

Write the GCL vertical volume flux `proj.omega` into `state.w` as a velocity (`w = ω / cell_area`, at
bottom faces k=1..nz+1; seabed and surface are ~0 by construction), matching the layout
`diagnose_vertical_velocity!` produces. Only ACTIVE columns are written (the Dirichlet-mouth `isopen`
cells are open boundaries whose ω is not continuity-closed — handled by the open-BC path, left w=0
here). Zeroes `state.w` elsewhere.
"""
function write_omega_velocity!(proj::BreathingProjector, grid::CurvilinearGrid, state::State)
    ng, nx, ny, nz = proj.ng, proj.nx, proj.ny, proj.nz
    w = state.w; fill!(w, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        inv_area = grid.pm[i+ng, j+ng] * grid.pn[i+ng, j+ng]
        for k in 1:nz+1
            w[i+ng, j+ng, k] = proj.omega[i, j, k] * inv_area
        end
    end
    return state
end

end # module ProjectionModule
