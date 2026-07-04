# src/BreathingTransportModule.jl
#
# OPT-IN breathing-sigma transport step: the two-time-level, volume-coordinate cascade that consumes
# the ProjectionModule's corrected transports + GCL ω and preserves C≡1 exactly on a breathing sigma
# grid (see the plan §1-2). This is the engine-side companion to the validated breathing-FFSL line
# kernel in HorizontalTransportModule and the implicit vertical solver here.
#
# Per adaptive sub-step [t, t+Δt] inside a hydro read, with the read's corrected transports held
# piecewise-constant (the projection targets the interval-mean ∂D̃/∂t, so linear-in-t volumes land
# exactly on the archived surface) and the departure volume taken at the sub-step's start fraction f0:
#   cascade volumes  Vn --divx(U*·Δt)--> V* --divy(U*·Δt)--> V** --divz(ω·Δt)--> V^{n+1}
#   x-sweep : breathing FFSL, departure Vn, arrival V*  (C^n     -> C*)
#   y-sweep : breathing FFSL, departure V*, arrival V** (C*      -> C**)
#   z-sweep : IMPLICIT upwind + backward-Euler diffusion, departure V**, arrival V^{n+1} (C** -> C^{n+1})
# Each divide uses the ARRIVAL intermediate volume; the volume increment and the tracer mass flux use
# the identical swept region. C≡1 is algebraically exact (the receptor holds 1.0). Lie split for now
# (forward); Strang symmetry is added with the reverse-time/adjoint work.

module BreathingTransportModule

export BreathingWork, build_breathing_work, breathing_transport!, breathing_courant, pad_transports!

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.ProjectionModule
using ..HydrodynamicTransport.HorizontalTransportModule: _ffsl_line_breathing!

"""
    BreathingWork

Per-simulation scratch for the breathing transport step (tracer-independent, reused every sub-step):
the padded corrected face transports (volume flux [m³/s]) and the cascade intermediate volumes.
Build once with [`build_breathing_work`](@ref); refresh the transports per hydro read with
[`pad_transports!`](@ref). `Uxf[f,jg,k]` = transport across the padded x-face f (between padded cells
f and f+1); `Uyf[ig,f,k]` = across the padded y-face f.
"""
mutable struct BreathingWork
    mx::Int; my::Int; nz::Int; ng::Int
    Uxf::Array{Float64,3}    # padded x-face transport [m³/s]
    Uyf::Array{Float64,3}    # padded y-face transport [m³/s]
    # cascade intermediate volumes. Lie (x→y→z): V0=Vn → V1 → V2. Strang (½x·½y·z·½y·½x):
    # V0=Vn → V1(½x) → V2(½y) → V3(z) → V4(½y) → V^{n+1}(½x).
    V0::Array{Float64,3}
    V1::Array{Float64,3}
    V2::Array{Float64,3}
    V3::Array{Float64,3}
    V4::Array{Float64,3}
end

function build_breathing_work(grid::CurvilinearGrid)
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
    mx, my = nx + 2ng, ny + 2ng
    z = () -> zeros(Float64, mx, my, nz)
    return BreathingWork(mx, my, nz, ng, z(), z(), z(), z(), z(), z(), z())
end

"""
    pad_transports!(bw, proj)

Copy the projector's corrected per-layer face transports (`proj.Ux`, `proj.Uy`, physical indexing)
into the padded engine arrays `bw.Uxf`, `bw.Uyf`. Call once per hydro read (after `project!`).
`Ux[i,j,k]` is the west face of physical cell i → padded x-face `i-1+ng`; `Uy[i,j,k]` the south face
of physical cell j → padded y-face `j-1+ng`.
"""
function pad_transports!(bw::BreathingWork, proj::BreathingProjector)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    fill!(bw.Uxf, 0.0); fill!(bw.Uyf, 0.0)
    @inbounds for k in 1:nz
        for j in 1:ny, i in 1:nx+1
            bw.Uxf[i-1+ng, j+ng, k] = proj.Ux[i, j, k]
        end
        for j in 1:ny+1, i in 1:nx
            bw.Uyf[i+ng, j-1+ng, k] = proj.Uy[i, j, k]
        end
    end
    return bw
end

# Tracer-independent Strang cascade volumes for the sub-step: V0=Vn(f0) → V1(½x) → V2(½y) → V3(z) →
# V4(½y) (V^{n+1}(½x) is recomputed inside the last sweep). Each `Vk = V_{k-1} − Δt_sweep·div(flux)` with
# Δt_sweep = dt/2 for the half x/y sweeps and dt for the full z sweep. Ghost/land/dry cells get V0=0 (the
# FFSL walk treats volume ≤ 0 as a wall + fills with ambient). Δσ_k from the projector; depth linear in
# the read. Total volume change telescopes to −dt·(divx+divy+divz) = geometric (Hnp−Hn), so C≡1 closes.
function _update_cascade_volumes!(bw::BreathingWork, proj::BreathingProjector, dt::Float64, f0::Float64)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    V0, V1, V2, V3, V4 = bw.V0, bw.V1, bw.V2, bw.V3, bw.V4
    Uxf, Uyf = bw.Uxf, bw.Uyf
    hdt = 0.5 * dt
    fill!(V0, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        proj.wet[i, j] || continue
        H = proj.Hn[i, j] + f0 * (proj.Hnp[i, j] - proj.Hn[i, j])
        a = proj.dxo[i, j] * proj.dyo[i, j]
        ig, jg = i + ng, j + ng
        for k in 1:nz; V0[ig, jg, k] = a * proj.dsig[k] * H; end
    end
    _breathe_x!(V1, V0, Uxf, hdt, bw.mx, bw.my, nz)   # ½x
    _breathe_y!(V2, V1, Uyf, hdt, bw.mx, bw.my, nz)   # ½y
    # z: V3 = V2 − dt·divz(ω) on active columns (elsewhere unchanged)
    copyto!(V3, V2)
    @inbounds for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        ig, jg = i + ng, j + ng
        for k in 1:nz; V3[ig, jg, k] = V2[ig, jg, k] - dt * (proj.omega[i, j, k+1] - proj.omega[i, j, k]); end
    end
    _breathe_y!(V4, V3, Uyf, hdt, bw.mx, bw.my, nz)   # ½y
    return nothing
end

# Vout = Vin − Δt·divx(Uxf):  divx at padded cell g = Uxf[g] − Uxf[g-1].
function _breathe_x!(Vout, Vin, Uxf, dtc, mx, my, nz)
    @inbounds for k in 1:nz, jg in 1:my
        Vout[1, jg, k] = Vin[1, jg, k]
        for g in 2:mx; Vout[g, jg, k] = Vin[g, jg, k] - dtc * (Uxf[g, jg, k] - Uxf[g-1, jg, k]); end
    end
end
# Vout = Vin − Δt·divy(Uyf):  divy at padded cell g = Uyf[.,g] − Uyf[.,g-1].
function _breathe_y!(Vout, Vin, Uyf, dtc, mx, my, nz)
    @inbounds for k in 1:nz, ig in 1:mx
        Vout[ig, 1, k] = Vin[ig, 1, k]
        for g in 2:my; Vout[ig, g, k] = Vin[ig, g, k] - dtc * (Uyf[ig, g, k] - Uyf[ig, g-1, k]); end
    end
end

# Implicit breathing vertical column solve (advection: backward-Euler upwind on the GCL ω; diffusion:
# backward-Euler implicit — 1st-order in time, more diffusive than Crank-Nicolson but Kz is tiny here).
# Departure volume `Vd` (= V**), arrival `Va` (= Vd − Δt·divz(ω)); `om[k]` = ω at the bottom face of
# cell k (om[1]=seabed=0, om[nz+1]=surface≈0). Solves A·Cnew = Vd·Cold in place (Thomas).
# Row-sum identity dd+dl+du = Vd ⇒ C≡1 exact (advection sums to Δt(ω[k+1]−ω[k]); diffusion sums to 0).
# The row-sum margin (hence the M-matrix / positivity condition) is the DEPARTURE volume Vd = V** > 0
# (NOT Va): off-diagonals are ≤ 0 and dd = Vd + Δt·max(ω[k],0) − Δt·min(ω[k+1],0) + Δt(gb+gt) ≥ Vd, so
# Vd > 0 ⇒ strictly diagonally dominant nonsingular M-matrix ⇒ A⁻¹ ≥ 0, unconditionally in Δt. V** > 0
# is enforced by the horizontal Courant bound in `breathing_courant` (f_x+f_y < Vn); Va > 0 holds
# geometrically (Va = a·Δσ·H(end fraction) ≥ a·Δσ·D_min).
@inline function _advdiff_z_col_breathing!(C, om, Vd, Va, area, dz, nz, dt, Kz, dl, dd, du, rhs, cprime)
    @inbounds begin
        for k in 1:nz
            wl = om[k]; wu = om[k+1]
            # implicit upwind advection contributions
            adv_diag = (wu >= 0.0 ? wu : 0.0) - (wl < 0.0 ? wl : 0.0)
            adv_sub  = (wl >= 0.0 ? -wl : 0.0)      # couples k-1
            adv_sup  = (wu <  0.0 ?  wu : 0.0)      # couples k+1
            # implicit diffusion conductances at bottom/top faces (0 at the seabed/surface)
            gb = (k > 1  && dz[k] > 0.0 && dz[k-1] > 0.0) ? Kz * area / (0.5*(dz[k]+dz[k-1])) : 0.0
            gt = (k < nz && dz[k] > 0.0 && dz[k+1] > 0.0) ? Kz * area / (0.5*(dz[k]+dz[k+1])) : 0.0
            dd[k] = Va[k] + dt*adv_diag + dt*(gb + gt)
            dl[k] = dt*adv_sub - dt*gb
            du[k] = dt*adv_sup - dt*gt
            rhs[k] = Vd[k] * C[k]
        end
        # Thomas solve A·x = rhs
        cprime[1] = du[1] / dd[1]; rhs[1] = rhs[1] / dd[1]
        for k in 2:nz
            m = dd[k] - dl[k]*cprime[k-1]
            cprime[k] = du[k] / m
            rhs[k] = (rhs[k] - dl[k]*rhs[k-1]) / m
        end
        C[nz] = rhs[nz]
        for k in nz-1:-1:1
            C[k] = rhs[k] - cprime[k]*C[k+1]
        end
    end
    return nothing
end

# One breathing FFSL x-sweep along every y-row (departure volume field `vdep`, swept volume Uxf·dtc,
# ambient camb). Updates C's physical x-cells in place. `s` = the per-task scratch tuple.
function _xsweep!(C, vdep, Uxf, dtc, camb, s, mx, my, ng, nx, nz)
    @inbounds for k in 1:nz, jg in 1:my
        for g in 1:mx; s.crow[g] = Float64(C[g, jg, k]); s.Srow[g] = Uxf[g, jg, k] * dtc; end
        _ffsl_line_breathing!(view(s.crow,1:mx), view(s.crow,1:mx), view(vdep,:,jg,k), view(s.varr,1:mx),
            view(s.Srow,1:mx), view(s.cL,1:mx), view(s.cR,1:mx), view(s.Flo,1:mx), view(s.Fhi,1:mx),
            view(s.Ctd,1:mx), view(s.Rp,1:mx), view(s.Rm,1:mx), mx, ng, nx; camb=camb)
        for ip in 1:nx; C[ip+ng, jg, k] = s.crow[ip+ng]; end
    end
end
# One breathing FFSL y-sweep along every x-column.
function _ysweep!(C, vdep, Uyf, dtc, camb, s, mx, my, ng, ny, nz)
    @inbounds for k in 1:nz, ig in 1:mx
        for g in 1:my; s.crow[g] = Float64(C[ig, g, k]); s.Srow[g] = Uyf[ig, g, k] * dtc; end
        _ffsl_line_breathing!(view(s.crow,1:my), view(s.crow,1:my), view(vdep,ig,:,k), view(s.varr,1:my),
            view(s.Srow,1:my), view(s.cL,1:my), view(s.cR,1:my), view(s.Flo,1:my), view(s.Fhi,1:my),
            view(s.Ctd,1:my), view(s.Rp,1:my), view(s.Rm,1:my), my, ng, ny; camb=camb)
        for jp in 1:ny; C[ig, jp+ng, k] = s.crow[jp+ng]; end
    end
end
# Implicit breathing vertical sweep over active columns (departure volume field `Vd3`).
function _zsweep!(C, proj, Vd3, dt, Kz, s, ng, nx, ny, nz)
    @inbounds for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        ig, jg = i + ng, j + ng
        a = proj.dxo[i, j] * proj.dyo[i, j]
        for k in 1:nz
            s.om[k] = proj.omega[i, j, k]; s.Vdc[k] = Vd3[ig, jg, k]
            s.dz[k] = s.Vdc[k] > 0.0 ? s.Vdc[k] / a : 0.0
        end
        s.om[nz+1] = proj.omega[i, j, nz+1]
        for k in 1:nz; s.Va[k] = s.Vdc[k] - dt * (s.om[k+1] - s.om[k]); end
        _advdiff_z_col_breathing!(view(C, ig, jg, :), s.om, s.Vdc, s.Va, a, s.dz, nz, dt, Kz,
                                  s.dl, s.dd, s.du, s.rhs, s.cprime)
    end
end

"""
    breathing_transport!(state, proj, bw, grid, dt, f0; camb=0.0, Kz=1e-4)

One breathing transport sub-step for every tracer: the STRANG-split volume cascade ½x·½y·z·½y·½x
(2nd-order + self-adjoint, the plan's recommended architecture). `dt` = the sub-step, `f0` = its start
fraction within the current hydro read (departure volumes are taken there). `camb` = the ambient
concentration filled where the FFSL departure region exits the domain / crosses a dry cell (1 for a
C≡1 check, 0 for the clean-ocean pathogen tracer). Mutates `state.tracers` in place.
"""
function breathing_transport!(state::State, proj::BreathingProjector, bw::BreathingWork,
                              grid::CurvilinearGrid, dt::Float64, f0::Float64;
                              camb::Float64=0.0, Kz::Float64=1e-4)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    mx, my = bw.mx, bw.my
    _update_cascade_volumes!(bw, proj, dt, f0)
    Uxf, Uyf = bw.Uxf, bw.Uyf
    V0, V1, V2, V3, V4 = bw.V0, bw.V1, bw.V2, bw.V3, bw.V4
    hdt = 0.5 * dt

    tracer_names = collect(keys(state.tracers))
    ntr = length(tracer_names)
    nchunks = max(1, min(Threads.nthreads(), ntr))
    Threads.@threads for cid in 1:nchunks
        mmax = max(mx, my)
        s = (crow=Vector{Float64}(undef, mmax), Srow=Vector{Float64}(undef, mmax),
             varr=Vector{Float64}(undef, mmax), cL=Vector{Float64}(undef, mmax), cR=Vector{Float64}(undef, mmax),
             Flo=Vector{Float64}(undef, mmax), Fhi=Vector{Float64}(undef, mmax), Ctd=Vector{Float64}(undef, mmax),
             Rp=Vector{Float64}(undef, mmax), Rm=Vector{Float64}(undef, mmax),
             dl=Vector{Float64}(undef, nz), dd=Vector{Float64}(undef, nz), du=Vector{Float64}(undef, nz),
             rhs=Vector{Float64}(undef, nz), cprime=Vector{Float64}(undef, nz), om=Vector{Float64}(undef, nz+1),
             Vdc=Vector{Float64}(undef, nz), Va=Vector{Float64}(undef, nz), dz=Vector{Float64}(undef, nz))
        ti = cid
        while ti <= ntr
            C = state.tracers[tracer_names[ti]]
            # Strang split ½x · ½y · z · ½y · ½x, threading the cascade departure volumes V0..V4.
            _xsweep!(C, V0, Uxf, hdt, camb, s, mx, my, ng, nx, nz)
            _ysweep!(C, V1, Uyf, hdt, camb, s, mx, my, ng, ny, nz)
            _zsweep!(C, proj, V2, dt, Kz, s, ng, nx, ny, nz)
            _ysweep!(C, V3, Uyf, hdt, camb, s, mx, my, ng, ny, nz)
            _xsweep!(C, V4, Uxf, hdt, camb, s, mx, my, ng, nx, nz)
            ti += nchunks
        end
    end
    return nothing
end

"""
    breathing_courant(proj, bw) -> Float64

Max advective Courant PER UNIT dt of the corrected transport over active cells: `max Σ(OUTGOING face
transports) / V_cell` (per layer). The controller picks `dt ≤ cfl_target/this` with cfl_target < 1, so
the TOTAL outgoing swept volume stays < cfl_target·V_cell — which guarantees every cascade intermediate
volume V*, V**, V^{n+1} stays positive (V* ≥ Vn−f_x, V** ≥ Vn−f_x−f_y, Va ≥ Vn−f_x−f_y−f_z, with
f_x+f_y+f_z = Δt·(total outgoing) < Vn). V_cell uses `min(Hn,Hnp)`, the smallest volume over the read.
"""
function breathing_courant(proj::BreathingProjector, bw::BreathingWork)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    Uxf, Uyf = bw.Uxf, bw.Uyf
    maxc = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        proj.active[i, j] || continue
        ig, jg = i + ng, j + ng
        a = proj.dxo[i, j] * proj.dyo[i, j]
        # smallest volume the cell reaches over the read (min-depth): the drying bound, so a
        # draining cell's late-read sub-step can't overshoot volume positivity (Claim 4).
        Hd = max(min(proj.Hn[i, j], proj.Hnp[i, j]), proj.D_min)
        for k in 1:nz
            out = max(Uxf[ig, jg, k], 0.0) + max(-Uxf[ig-1, jg, k], 0.0) +
                  max(Uyf[ig, jg, k], 0.0) + max(-Uyf[ig, jg-1, k], 0.0) +
                  max(proj.omega[i, j, k+1], 0.0) + max(-proj.omega[i, j, k], 0.0)
            Vk = a * proj.dsig[k] * Hd
            Vk > 0.0 && (maxc = max(maxc, out / Vk))
        end
    end
    return maxc
end

end # module BreathingTransportModule
