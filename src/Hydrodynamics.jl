# src/Hydrodynamics.jl

module HydrodynamicsModule

export update_hydrodynamics!, update_hydrodynamics_placeholder!, diagnose_vertical_velocity!

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.ProjectionModule
using ..HydrodynamicTransport.GridModule: rebuild_metrics!
using NCDatasets
using Dates
using Base.Threads: @threads

# --- The placeholder functions for CartesianGrid remain unchanged ---
function update_hydrodynamics_placeholder!(state::State, grid::CartesianGrid, time::Float64)
    ng = grid.ng
    nx, ny, nz = grid.dims
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    
    dx = (grid.x[ng+2, ng+1, 1] - grid.x[ng+1, ng+1, 1])
    dy = (grid.y[ng+1, ng+2, 1] - grid.y[ng+1, ng+1, 1])
    Lx = nx * dx; Ly = ny * dy
    center_x = Lx / 2; center_y = Ly / 2
    period = 200.0; omega = 2π / period
    
    for k in 1:nz, j_glob in 1:ny_tot, i_glob in 1:nx_tot+1
        i_phys_face = i_glob - ng - 0.5; j_phys_center = j_glob - ng
        x_coord_face = i_phys_face * dx; y_coord_center = (j_phys_center - 0.5) * dy
        rx = x_coord_face - center_x; ry = y_coord_center - center_y
        state.u[i_glob, j_glob, k] = -omega * ry
    end

    for k in 1:nz, j_glob in 1:ny_tot+1, i_glob in 1:nx_tot
        i_phys_center = i_glob - ng; j_phys_face = j_glob - ng - 0.5
        x_coord_center = (i_phys_center - 0.5) * dx; y_coord_face = j_phys_face * dy
        rx = x_coord_center - center_x; ry = y_coord_face - center_y
        state.v[i_glob, j_glob, k] = omega * rx
    end
    state.w .= 0.0
    state.zeta .= 0.0
end

function update_hydrodynamics_placeholder!(state::State, grid::CurvilinearGrid, time::Float64)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    dx_approx = 1.0 / grid.pm[ng+nx÷2, ng+ny÷2]
    dy_approx = 1.0 / grid.pn[ng+nx÷2, ng+ny÷2]

    Lx = nx * dx_approx; Ly = ny * dy_approx
    center_x = Lx / 2; center_y = Ly / 2
    omega = 0.001

    # Set all velocities to zero initially
    state.u .= 0.0
    state.v .= 0.0

    # Calculate velocities for INTERIOR faces only
    for k in 1:nz
        # U-velocities (X-faces)
        for j_phys in 1:ny, i_phys in 2:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            y_center = (j_phys - 0.5) * dy_approx
            # FIX: Corrected sign for counter-clockwise rotation
            state.u[i_glob, j_glob, k] = omega * (y_center - center_y)
        end
        # V-velocities (Y-faces)
        for j_phys in 2:ny, i_phys in 1:nx
            i_glob, j_glob = i_phys + ng, j_phys + ng
            x_center = (i_phys - 0.5) * dx_approx
            state.v[i_glob, j_glob, k] = -omega * (x_center - center_x)
        end
    end
    
    state.w .= 0.0
    state.zeta .= 0.0
end


# Load (and coalesce) the bracketing time-slab for `idx` into the cache, but only if it
# isn't already cached. This is what removes the per-timestep NetCDF reads.
function _ensure_hydro_slab!(cache::HydroSlabCache, ds::NCDataset, hydro_data::HydrodynamicData, fields, idx::Int)
    haskey(cache.slabs, idx) && return
    d = Dict{Symbol, Array{Float64}}()
    for (_state_field, standard_name, has_z) in fields
        if haskey(hydro_data.var_map, standard_name)
            nc_var_name = hydro_data.var_map[standard_name]
            if haskey(ds, nc_var_name)
                d[standard_name] = has_z ? coalesce.(ds[nc_var_name][:, :, :, idx], 0.0) :
                                           coalesce.(ds[nc_var_name][:, :, idx], 0.0)
            end
        end
    end
    cache.slabs[idx] = d
    return
end

# --- Real-data hydrodynamics with temporal interpolation + cached time-slabs ---
# Numerically identical to the previous version; the only change is that the time axis is
# converted once and the bracketing slabs are read from disk only when their time index
# changes (then kept in hydro_data.cache), instead of being re-read every timestep.
"""
    diagnose_vertical_velocity!(state, grid)

Diagnose the sigma-coordinate vertical velocity from continuity, using the horizontal transports
already in `state.u`/`state.v` (e.g. MARS3D `UZ`/`VZ`). Many hydro files store no `w`/omega, so
the offline transport would otherwise run with **no vertical advection** — and on a sigma grid the
horizontal sweeps alone do not preserve a uniform tracer. This fills `state.w` with the vertical
velocity that closes the discrete volume budget per cell.

For each wet column, the horizontal volume divergence of cell k is
`HDiv(k) = u·Ax|_east - u·Ax|_west + v·Ay|_north - v·Ay|_south`, and the vertical volume flux
`Wflux` obeys `Wflux(k+1) = Wflux(k) - HDiv(k)`. Rigid-lid closure pins `Wflux = 0` at the seabed
and the surface; the residual column divergence `Dtot` (the neglected sea-surface-height tendency,
since cell volumes are static) is distributed by layer thickness:
`Wflux(k) = -Σ_{l<k} HDiv(l) + (H_below(k)/H_total)·Dtot`, then `w = Wflux / cell_area`. `w` is stored
at cell **bottom faces** (`w[k]` = face between cells k-1 and k), matching `vertical_transport!`;
`w[1]` (seabed) and `w[nz+1]` (surface) are 0. Land columns get `w = 0`.
"""
function diagnose_vertical_velocity!(state::State, grid::CurvilinearGrid)
    ng = grid.ng; nx, ny, nz = grid.nx, grid.ny, grid.nz
    u = state.u; v = state.v; w = state.w
    fax = grid.face_area_x; fay = grid.face_area_y; pm = grid.pm; pn = grid.pn; vol = grid.volume
    # Horizontal volume divergence of cell (ig,jg,k). Cheap; recomputed (not stored) so the loop
    # needs no per-task scratch (avoids the threadid()/thread-pool pitfall) and zero allocation.
    @inline hdiv(ig, jg, k) = @inbounds(u[ig+1, jg, k] * fax[ig+1, jg, k] - u[ig, jg, k] * fax[ig, jg, k] +
                                        v[ig, jg+1, k] * fay[ig, jg+1, k] - v[ig, jg, k] * fay[ig, jg, k])
    @threads for j in 1:ny
        for i in 1:nx
            ig, jg = i + ng, j + ng
            @inbounds begin
                # Diagnose only at interior cells whose four horizontal faces all carry physical
                # flow. At coastline cells (a land/masked neighbour) or domain-edge cells the
                # horizontal continuity is broken (a face is cut, or the boundary inflow is unknown),
                # so the column "divergence" is an artefact -> diagnosing w there pumps spurious
                # vertical transport. Leave w = 0 there (no vertical advection in the boundary ring).
                edge = i == 1 || i == nx || j == 1 || j == ny
                if !grid.mask_rho[ig, jg] || edge ||
                   !grid.mask_rho[ig-1, jg] || !grid.mask_rho[ig+1, jg] ||
                   !grid.mask_rho[ig, jg-1] || !grid.mask_rho[ig, jg+1]
                    for k in 1:nz+1; w[ig, jg, k] = 0.0; end
                    continue
                end
                inv_area = pm[ig, jg] * pn[ig, jg]          # 1 / cell horizontal area
                # Pass 1: column-total horizontal divergence and depth.
                Dtot = 0.0; Htot = 0.0
                for k in 1:nz
                    Dtot += hdiv(ig, jg, k); Htot += vol[ig, jg, k] * inv_area
                end
                invH = Htot > 0.0 ? 1.0 / Htot : 0.0
                # Pass 2: cumulative flux with rigid-lid closure (w = 0 at seabed and surface; the
                # residual column divergence Dtot is spread by layer thickness).
                w[ig, jg, 1] = 0.0; w[ig, jg, nz+1] = 0.0
                cum_div = 0.0; cum_h = 0.0
                for k in 2:nz
                    cum_div += hdiv(ig, jg, k-1); cum_h += vol[ig, jg, k-1] * inv_area
                    Wflux = -cum_div + (cum_h * invH) * Dtot
                    w[ig, jg, k] = Wflux * inv_area              # w = Wflux / area
                end
            end
        end
    end
    return nothing
end

function update_hydrodynamics!(state::State, grid::CurvilinearGrid, ds::NCDataset, hydro_data::HydrodynamicData, time::Float64;
                               diagnose_w::Bool=true, projector::Union{Nothing,BreathingProjector}=nothing,
                               reverse::Bool=false)
    ng = grid.ng
    cache = hydro_data.cache
    breathing = projector !== nothing

    # Convert the time axis once, then reuse from the cache.
    if cache.time_seconds === nothing
        time_var_name = get(hydro_data.var_map, :time, "time")
        time_dim_raw = ds[time_var_name][:]
        cache.time_seconds = if eltype(time_dim_raw) <: DateTime
            t0 = time_dim_raw[1]; [(dt - t0).value / 1000.0 for dt in time_dim_raw]
        else
            Float64.(time_dim_raw)
        end
    end
    time_dim_seconds = cache.time_seconds
    n_times = length(time_dim_seconds)

    local idx1, idx2, weight
    if time <= time_dim_seconds[1]; idx1 = 1; idx2 = 1; weight = 0.0
    elseif time >= time_dim_seconds[n_times]; idx1 = n_times; idx2 = n_times; weight = 0.0
    else
        idx1 = searchsortedlast(time_dim_seconds, time); idx2 = idx1 + 1
        # REVERSE-TIME read selection at a boundary: a descending pass that lands EXACTLY on a read
        # boundary tv[r] must use the read it is about to traverse DOWNWARD, [tv[r-1], tv[r]], not the read
        # [tv[r], tv[r+1]] that searchsortedlast returns (which starts at tv[r]). Otherwise the sub-step
        # just below each read boundary advects with the WRONG (adjacent) read's transports — an O(dt)
        # per-read-boundary reverse-time non-mirror that is the real reverse-time reciprocity floor.
        # Forward (reverse=false) is untouched: it traverses [tv[r], tv[r+1]] UPWARD, so tv[r] is correct.
        if reverse && idx1 > 1 && (time - time_dim_seconds[idx1]) <= 1e-6 * (time_dim_seconds[idx1+1] - time_dim_seconds[idx1])
            idx1 -= 1; idx2 = idx1 + 1
        end
        t1 = time_dim_seconds[idx1]; t2 = time_dim_seconds[idx2]; time_interval = t2 - t1
        weight = (time_interval > 1e-9) ? (time - t1) / time_interval : 0.0
    end

    # (state_field, standard_name, has_z). zeta is a 2-D field broadcast across z layers.
    fields = ((state.u, :u, true), (state.v, :v, true), (state.temperature, :temp, true),
              (state.salinity, :salt, true), (state.zeta, :zeta, false))

    # Read bracketing slabs from disk only if not already cached, then bound cache size.
    _ensure_hydro_slab!(cache, ds, hydro_data, fields, idx1)
    idx2 != idx1 && _ensure_hydro_slab!(cache, ds, hydro_data, fields, idx2)
    if length(cache.slabs) > 4
        for k in collect(keys(cache.slabs)); (k == idx1 || k == idx2) || delete!(cache.slabs, k); end
    end

    slabs1 = cache.slabs[idx1]; slabs2 = cache.slabs[idx2]
    for (state_field, standard_name, has_z) in fields
        haskey(slabs1, standard_name) || continue
        # Rigid-lid default keeps state.zeta ≡ 0 (bit-identical: the transport kernels read `h + zeta`
        # face depths, so populating it would change the frozen-volume results). Breathing mode reads
        # the true free surface into state.zeta.
        (standard_name === :zeta && !breathing) && continue
        data_slice1 = slabs1[standard_name]
        if has_z
            nx_phys, ny_phys, nz_phys = size(data_slice1)
            interior_view = view(state_field, ng+1:nx_phys+ng, ng+1:ny_phys+ng, 1:nz_phys)
            if weight > 1e-9
                interior_view .= (1.0 - weight) .* data_slice1 .+ weight .* slabs2[standard_name]
            else
                interior_view .= data_slice1
            end
        else
            nx_phys, ny_phys = size(data_slice1)
            interior_view_3d = view(state_field, ng+1:nx_phys+ng, ng+1:ny_phys+ng, :)
            interpolated_data_2d = (weight > 1e-9) ?
                ((1.0 - weight) .* data_slice1 .+ weight .* slabs2[standard_name]) : data_slice1
            for k in 1:size(interior_view_3d, 3)
                view(interior_view_3d, :, :, k) .= interpolated_data_2d
            end
        end
    end
    if breathing
        # OPT-IN breathing-sigma continuity correction: solve the barotropic projection once per hydro
        # read, breathe the sigma metric to the free surface, and set the GCL ω (replaces the rigid-lid
        # w-diagnosis). See ProjectionModule + the plan.
        _breathing_update!(state, grid, projector, cache, idx1, idx2, time_dim_seconds; reverse=reverse)
    elseif diagnose_w
        # Files store no w/omega -> diagnose the vertical velocity from continuity so the sigma
        # transport has vertical advection (and preserves a uniform tracer). Disable with diagnose_w=false.
        diagnose_vertical_velocity!(state, grid)
    end
    return nothing
end

# Breathing-sigma per-read update: project the barotropic transport onto discrete continuity vs the
# free surface, rebuild the sigma metric so volumes breathe, and write the GCL vertical flux ω into
# state.w. Solved ONCE per hydro bracket (cadence guard `projector.last_idx`); intermediate adaptive
# steps and rejected trials reuse the cached projection. At the clamped file ends (idx2==idx1) there
# is no interval, so the previous projection is retained.
function _breathing_update!(state::State, grid::CurvilinearGrid, projector::BreathingProjector,
                            cache::HydroSlabCache, idx1::Int, idx2::Int, tsec::Vector{Float64};
                            reverse::Bool=false)
    ng = grid.ng
    slabs1 = cache.slabs[idx1]; slabs2 = cache.slabs[idx2]
    haskey(slabs1, :zeta) ||
        error("breathing mode requires a free surface (zeta/XE) in the hydro file; none detected")
    if idx2 != idx1 && projector.last_idx != idx1
        # REVERSE-TIME / ADJOINT: negate the velocity and SWAP the bracketing surfaces (η_n↔η_np1). By
        # the linearity of the projection (vetted: U*(−u,−∂η/∂t)=−U* exactly, and the D̃-weighted Poisson
        # is self-adjoint), this yields −U*, −ω and the reversed volume chain (Hn↔Hnp) — which makes the
        # breathing cascade the EXACT discrete adjoint of the forward step (machine precision for the
        # unlimited linear scheme; the FCT limiter + wet/dry are the only reciprocity floors).
        eta_n, eta_np1 = reverse ? (slabs2[:zeta], slabs1[:zeta]) : (slabs1[:zeta], slabs2[:zeta])
        # Project the read using the MID-read velocity (mean of the two bracket slabs), NOT the current
        # sub-step interpolation. This makes the forward pass (which enters the read near its start) and
        # the reverse pass (which enters near its end) project the IDENTICAL transport — required for the
        # reverse step to be the exact adjoint (agent-flagged transport-replay condition). C≡1 is
        # velocity-independent, so this is safe; it is also more accurate (interval-representative).
        if haskey(slabs1, :u) && haskey(slabs2, :u)
            um = 0.5 .* (slabs1[:u] .+ slabs2[:u]); nxu, nyu, nzu = size(um)
            view(state.u, ng+1:nxu+ng, ng+1:nyu+ng, 1:nzu) .= um
        end
        if haskey(slabs1, :v) && haskey(slabs2, :v)
            vm = 0.5 .* (slabs1[:v] .+ slabs2[:v]); nxv, nyv, nzv = size(vm)
            view(state.v, ng+1:nxv+ng, ng+1:nyv+ng, 1:nzv) .= vm
        end
        if reverse
            @. state.u = -state.u; @. state.v = -state.v
        end
        dt = tsec[idx2] - tsec[idx1]
        project!(projector, grid, state.u, state.v, eta_n, eta_np1, dt)
        rebuild_metrics!(grid, padded_depth!(projector, grid); d_floor=1.0)
        write_omega_velocity!(projector, grid, state)
        projector.last_idx = idx1
        projector.t_read_start = tsec[idx1]; projector.t_read_end = tsec[idx2]
    end
    return nothing
end

end # module HydrodynamicsModule

