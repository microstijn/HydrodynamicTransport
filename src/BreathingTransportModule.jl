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
    Vn::Array{Float64,3}     # departure volumes (padded), per sub-step
    Vstar::Array{Float64,3}  # after the x-sweep
    Vss::Array{Float64,3}    # after the y-sweep
end

function build_breathing_work(grid::CurvilinearGrid)
    ng, nx, ny, nz = grid.ng, grid.nx, grid.ny, grid.nz
    mx, my = nx + 2ng, ny + 2ng
    z = () -> zeros(Float64, mx, my, nz)
    return BreathingWork(mx, my, nz, ng, z(), z(), z(), z(), z())
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

# Departure volumes at sub-step start fraction f0, and the tracer-independent cascade intermediates
# V* = Vn − divx(U*·Δt), V** = V* − divy(U*·Δt). Ghost/land/dry cells get 0 (the FFSL walk treats
# volume ≤ 0 as a wall + fills with ambient). Δσ_k from the projector; depth linear in the read.
function _update_cascade_volumes!(bw::BreathingWork, proj::BreathingProjector, dt::Float64, f0::Float64)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    Vn, Vstar, Vss = bw.Vn, bw.Vstar, bw.Vss
    Uxf, Uyf = bw.Uxf, bw.Uyf
    fill!(Vn, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        proj.wet[i, j] || continue
        H = proj.Hn[i, j] + f0 * (proj.Hnp[i, j] - proj.Hn[i, j])
        a = proj.dxo[i, j] * proj.dyo[i, j]
        ig, jg = i + ng, j + ng
        for k in 1:nz; Vn[ig, jg, k] = a * proj.dsig[k] * H; end
    end
    # V* = Vn − dt·divx(U*):  divx at padded cell g = Uxf[g] − Uxf[g-1]
    @inbounds for k in 1:nz, jg in 1:bw.my
        Vstar[1, jg, k] = Vn[1, jg, k]
        for g in 2:bw.mx
            Vstar[g, jg, k] = Vn[g, jg, k] - dt * (Uxf[g, jg, k] - Uxf[g-1, jg, k])
        end
    end
    # V** = V* − dt·divy(U*):  divy at padded cell g = Uyf[.,g] − Uyf[.,g-1]
    @inbounds for k in 1:nz, ig in 1:bw.mx
        Vss[ig, 1, k] = Vstar[ig, 1, k]
        for g in 2:bw.my
            Vss[ig, g, k] = Vstar[ig, g, k] - dt * (Uyf[ig, g, k] - Uyf[ig, g-1, k])
        end
    end
    return nothing
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

"""
    breathing_transport!(state, proj, bw, grid, dt, f0; camb=0.0, Kz=1e-4)

One breathing transport sub-step for every tracer: the x→y→z volume cascade (Lie split). `dt` = the
sub-step, `f0` = its start fraction within the current hydro read (departure volumes are taken there).
`camb` is the ambient concentration filled where the FFSL departure region exits the domain / crosses
a dry cell (1 for a C≡1 check, 0 for the clean-ocean pathogen tracer). Mutates `state.tracers` in place.
"""
function breathing_transport!(state::State, proj::BreathingProjector, bw::BreathingWork,
                              grid::CurvilinearGrid, dt::Float64, f0::Float64;
                              camb::Float64=0.0, Kz::Float64=1e-4)
    ng, nx, ny, nz = bw.ng, proj.nx, proj.ny, proj.nz
    mx, my = bw.mx, bw.my
    _update_cascade_volumes!(bw, proj, dt, f0)
    Uxf, Uyf, Vn, Vstar, Vss = bw.Uxf, bw.Uyf, bw.Vn, bw.Vstar, bw.Vss

    tracer_names = collect(keys(state.tracers))
    ntr = length(tracer_names)
    nchunks = max(1, min(Threads.nthreads(), ntr))
    Threads.@threads for cid in 1:nchunks
        # per-task scratch
        mmax = max(mx, my)
        crow = Vector{Float64}(undef, mmax); Srow = Vector{Float64}(undef, mmax)
        varr = Vector{Float64}(undef, mmax); cL = Vector{Float64}(undef, mmax); cR = Vector{Float64}(undef, mmax)
        Flo = Vector{Float64}(undef, mmax); Fhi = Vector{Float64}(undef, mmax); Ctd = Vector{Float64}(undef, mmax)
        Rp = Vector{Float64}(undef, mmax); Rm = Vector{Float64}(undef, mmax)
        dl = Vector{Float64}(undef, nz); dd = Vector{Float64}(undef, nz); du = Vector{Float64}(undef, nz)
        rhs = Vector{Float64}(undef, nz); cprime = Vector{Float64}(undef, nz)
        om = Vector{Float64}(undef, nz+1); Vd = Vector{Float64}(undef, nz); Va = Vector{Float64}(undef, nz)
        dz = Vector{Float64}(undef, nz)
        ti = cid
        while ti <= ntr
            C = state.tracers[tracer_names[ti]]
            # x-sweep: departure Vn -> arrival V* (recomputed internally, == bw.Vstar)
            @inbounds for k in 1:nz, jg in 1:my
                for g in 1:mx; crow[g] = Float64(C[g, jg, k]); Srow[g] = Uxf[g, jg, k]*dt; end
                _ffsl_line_breathing!(view(crow,1:mx), view(crow,1:mx), view(Vn,:,jg,k), view(varr,1:mx),
                    view(Srow,1:mx), view(cL,1:mx), view(cR,1:mx), view(Flo,1:mx), view(Fhi,1:mx),
                    view(Ctd,1:mx), view(Rp,1:mx), view(Rm,1:mx), mx, ng, nx; camb=camb)
                for ip in 1:nx; C[ip+ng, jg, k] = crow[ip+ng]; end
            end
            # y-sweep: departure V* -> arrival V**
            @inbounds for k in 1:nz, ig in 1:mx
                for g in 1:my; crow[g] = Float64(C[ig, g, k]); Srow[g] = Uyf[ig, g, k]*dt; end
                _ffsl_line_breathing!(view(crow,1:my), view(crow,1:my), view(Vstar,ig,:,k), view(varr,1:my),
                    view(Srow,1:my), view(cL,1:my), view(cR,1:my), view(Flo,1:my), view(Fhi,1:my),
                    view(Ctd,1:my), view(Rp,1:my), view(Rm,1:my), my, ng, ny; camb=camb)
                for jp in 1:ny; C[ig, jp+ng, k] = crow[jp+ng]; end
            end
            # z-sweep: implicit breathing vertical, departure V** -> arrival V^{n+1}
            @inbounds for j in 1:ny, i in 1:nx
                proj.active[i, j] || continue
                ig, jg = i + ng, j + ng
                a = proj.dxo[i, j] * proj.dyo[i, j]
                for k in 1:nz
                    om[k] = proj.omega[i, j, k]
                    Vd[k] = Vss[ig, jg, k]
                    dz[k] = Vd[k] > 0.0 ? Vd[k] / a : 0.0
                end
                om[nz+1] = proj.omega[i, j, nz+1]
                for k in 1:nz; Va[k] = Vd[k] - dt*(om[k+1] - om[k]); end
                ccol = view(C, ig, jg, :)
                _advdiff_z_col_breathing!(ccol, om, Vd, Va, a, dz, nz, dt, Kz, dl, dd, du, rhs, cprime)
            end
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
