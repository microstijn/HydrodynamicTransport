# src/TimeSteppingModule.jl

module TimeSteppingModule

export run_simulation, run_and_store_simulation

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.HydrodynamicsModule
using ..HydrodynamicTransport.HorizontalTransportModule
using ..HydrodynamicTransport.VerticalTransportModule
using ..HydrodynamicTransport.ProjectionModule: BreathingProjector, build_projector
using ..HydrodynamicTransport.BreathingTransportModule
using ..HydrodynamicTransport.SourceSinkModule
using ..HydrodynamicTransport.BoundaryConditionsModule
using ..HydrodynamicTransport.SettlingModule
using ..HydrodynamicTransport.BedExchangeModule
using ..HydrodynamicTransport.OysterModule
using ..HydrodynamicTransport.ReceptorMonitoringModule: ReceptorMonitor, write_receptor_monitor!, flush_receptor_monitor!
using ..HydrodynamicTransport.UtilsModule: calculate_max_cfl_term, calculate_max_gradient_cfl_term
using ..HydrodynamicTransport.FluxLimitersModule 
using ProgressMeter
using NCDatasets
using JLD2

# --- Import the NEW TVD functions ---
using ..HorizontalTransportModule: advect_diffuse_tvd_implicit_x!, advect_diffuse_tvd_implicit_y!
using ..VerticalTransportModule: advect_diffuse_tvd_implicit_z!

# Copy the *mutable* dynamic fields of `src` into the preallocated `dst` (no allocation).
# Used in place of a per-timestep deepcopy: `dst` is a reusable trial buffer that, after a
# successful step, is swapped with `state` in O(1). Scratch buffers (_buffer1/_buffer2,
# flux_x/y/z) are intentionally not copied — they are overwritten before being read each step.
function _copy_dynamic_state!(dst::State, src::State)
    for (k, v) in src.tracers; copyto!(dst.tracers[k], v); end
    copyto!(dst.u, src.u); copyto!(dst.v, src.v); copyto!(dst.w, src.w); copyto!(dst.zeta, src.zeta)
    copyto!(dst.temperature, src.temperature); copyto!(dst.salinity, src.salinity)
    copyto!(dst.tss, src.tss); copyto!(dst.uvb, src.uvb)
    for (k, v) in src.bed_mass; copyto!(dst.bed_mass[k], v); end
    dst.time = src.time
    return dst
end

"""
    run_simulation(grid, initial_state, sources, start_time, end_time, dt; ...)

Main driver for running a hydrodynamic transport simulation.
... (docstring arguments) ...
- `limiter_func::Function`: The flux limiter function to use (e._g., `van_leer`, `minmod`).
"""
function run_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64;
                        ds::Union{NCDataset, Nothing}=nothing,
                        hydro_data::Union{HydrodynamicData, Nothing}=nothing,
                        use_adaptive_dt::Bool=false,
                        cfl_max::Float64=0.8,
                        dt_max::Float64=dt,
                        dt_min::Float64=0.1,
                        dt_growth_factor::Float64=1.1,
                        boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                        functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                        sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                        virtual_oysters::Vector{VirtualOyster}=Vector{VirtualOyster}(),
                        oyster_tracers::NamedTuple=NamedTuple(),
                        advection_scheme::Symbol=:FFSL,   # conservative + positive + peak-preserving; ~1.26x TVD here
                        limiter_func::Function=FluxLimitersModule.van_leer, # <-- NEW ARGUMENT
                        Kh::Float64=1.0,
                        Kz::Float64=1e-4,
                        D_crit::Float64=0.0,
                        # --- OPT-IN breathing-sigma continuity correction (default false = rigid-lid, bit-identical).
                        # Needs real hydro data (ds+hydro_data) with a free surface (zeta/XE). Replaces the transport
                        # step with the projected breathing cascade; the adaptive dt is driven by the corrected-
                        # transport Courant + read-boundary clip. `breathing_camb` = ambient tracer concentration
                        # filled where the departure region exits the domain / crosses dry cells (0 = clean ocean).
                        breathing::Bool=false,
                        breathing_camb::Float64=0.0,
                        diagnose_vertical_velocity::Bool=true,  # diagnose omega from continuity when files lack w
                        # --- BACKWARD / ADJOINT mode (opt-in; forward path byte-identical when false) ---
                        # For a LINEAR passive tracer the adjoint transport is the same advection-diffusion
                        # equation on the TIME-REVERSED, velocity-FLIPPED flow (flux-form + Crank-Nicolson are
                        # self-adjoint under u -> -u). Seeded at the receptor, one such run gives the receptor
                        # footprint / sensitivity to ALL sources at once (cf. FLEXPART backward mode). The
                        # internal clock `time` then measures LAG since the receptor pulse and increases 0..horizon
                        # (so output/monitor/adaptive-dt logic is unchanged); the real hydro time runs the other
                        # way, t_real = reverse_time_origin - time, and u,v,w are negated after each hydro read.
                        # NOTE: timed PointSources are not real-time-remapped here (PoC seeds via the receptor IC);
                        # release at the receptor as an initial condition or a lag-0 source.
                        reverse_time::Bool=false,
                        reverse_time_origin::Float64=0.0,  # real hydro time corresponding to lag 0 (window end)
                        output_dir::Union{String, Nothing}=nothing,
                        output_interval::Union{Float64, Nothing}=nothing,
                        write_full_state::Bool=true,
                        tracer_only_output::Bool=false,  # save only `state.tracers` (+ time); drop hydro/scratch. Analysis-only, NOT a restart checkpoint.
                        full_state_output_interval::Union{Float64, Nothing}=output_interval,
                        receptor_monitors::Vector{ReceptorMonitor}=ReceptorMonitor[],
                        receptor_monitor_interval::Union{Float64, Nothing}=output_interval,
                        restart_from::Union{String, Nothing}=nothing)

    local state_to_run, effective_start_time
    if restart_from !== nothing
        @info "Restarting simulation from checkpoint: $restart_from"
        restart_data = JLD2.load(restart_from)
        state_to_run = restart_data["state"]
        virtual_oysters = get(restart_data, "virtual_oysters", virtual_oysters)
        effective_start_time = state_to_run.time
    else
        state_to_run = initial_state
        effective_start_time = start_time
    end

    state = deepcopy(state_to_run)
    work = deepcopy(state)            # reusable trial buffer (replaces per-step deepcopy)
    time = effective_start_time
    current_dt = dt

    # Breathing-sigma: build the projection workspace once (needs a curvilinear grid + real hydro).
    local breathing_proj, breathing_work
    last_padded_idx = -1
    if breathing
        (grid isa CurvilinearGrid) || error("breathing mode requires a CurvilinearGrid")
        (ds !== nothing && hydro_data !== nothing) ||
            error("breathing mode requires ds + hydro_data (real hydro with a free surface)")
        breathing_proj = build_projector(grid)
        breathing_work = build_breathing_work(grid)
    end

    min_dt_taken = Inf
    max_dt_taken = 0.0
    start_wall_time = time_ns()

    # First output boundary STRICTLY after `time`. The `floor(...)+1` form is robust to
    # floating-point magnitude (an absolute `+1e-9` nudge is silently lost when `time` is a
    # large exact multiple of the interval, which collapsed dt_bound to 0 and aborted the run
    # before any output — fixed here).
    next_full_state_output_time = if full_state_output_interval !== nothing
        (floor(time / full_state_output_interval) + 1) * full_state_output_interval
    else
        Inf
    end

    next_receptor_monitor_time = if receptor_monitor_interval !== nothing
        (floor(time / receptor_monitor_interval) + 1) * receptor_monitor_interval
    else
        Inf
    end

    if output_dir !== nothing && write_full_state
        mkpath(output_dir)
    end
    
    desc_str = (ds !== nothing) ? "Simulating..." : "Simulating (test mode)..."
    pbar = Progress(floor(Int, end_time - time); desc=desc_str, dt=1.0)

    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt

        # Genuine CFL collapse guard: the stability-limited dt itself is below the floor.
        # Checked BEFORE the output/end-boundary clamp, because a step deliberately shortened
        # to land on a save boundary (or on end_time) is legitimately allowed to be < dt_min and
        # must not abort the run (previously this false-triggered at end-of-run and whenever the
        # output-boundary rounding collapsed the bound to 0).
        if use_adaptive_dt && trial_dt < dt_min
            println("\nWarning: CFL-limited timestep below dt_min. Stopping simulation.")
            break
        end

        # Consider both output clocks for finding the next dt bound
        dt_bound = end_time - time
        if write_full_state && next_full_state_output_time < Inf
            dt_bound = min(dt_bound, next_full_state_output_time - time)
        end
        if !isempty(receptor_monitors) && next_receptor_monitor_time < Inf
            dt_bound = min(dt_bound, next_receptor_monitor_time - time)
        end

        # Boundary clamp may legitimately make the step < dt_min (to hit a save/end time).
        trial_dt = min(trial_dt, dt_bound)
        if trial_dt < 1e-9; break; end

        if breathing
            # --- OPT-IN breathing-sigma step (no retry; dt chosen safe upfront). ---
            _copy_dynamic_state!(work, state)
            apply_boundary_conditions!(work, grid, boundary_conditions)
            # Project for the read containing the REAL hydro time (per-read-constant transports); breathe
            # the metric + set ω. In reverse-time mode the internal clock `time` is the LAG since the
            # receptor pulse, so the real hydro time counts backward: htime = origin − time, and the
            # projection negates u,v + swaps η (adjoint pass). Solved once per read (cadence guard inside).
            htime = reverse_time ? reverse_time_origin - time : time
            update_hydrodynamics!(work, grid, ds, hydro_data, htime; diagnose_w=false,
                                  projector=breathing_proj, reverse=reverse_time)
            if breathing_proj.last_idx != last_padded_idx
                pad_transports!(breathing_work, breathing_proj)
                last_padded_idx = breathing_proj.last_idx
            end
            # Adaptive dt: the corrected-transport Courant keeps the cascade volumes positive; clip to
            # the hydro-read boundary (stay within one projection) and the output/end bound (dt_bound).
            cour = breathing_courant(breathing_proj, breathing_work)
            dt_safe = cour > 0.0 ? cfl_max / cour : dt_max
            trial_dt = min(trial_dt, dt_safe)
            # Read-boundary clip: forward can't pass t_read_end; reverse can't pass t_read_start (htime
            # decreases as the lag grows).
            if reverse_time
                htime > breathing_proj.t_read_start && (trial_dt = min(trial_dt, htime - breathing_proj.t_read_start))
            else
                breathing_proj.t_read_end > time && (trial_dt = min(trial_dt, breathing_proj.t_read_end - time))
            end
            if trial_dt < 1e-9; break; end
            # Sub-step start fraction within the read. Reverse mode walks the read backward, so f0 mirrors:
            # forward f0 = (htime − t_start)/ΔT; reverse f0 = (t_end − htime)/ΔT (departure volume at htime).
            dT_read = breathing_proj.t_read_end - breathing_proj.t_read_start
            f0 = if dT_read <= 0.0
                0.0
            elseif reverse_time
                (breathing_proj.t_read_end - htime) / dT_read
            else
                (htime - breathing_proj.t_read_start) / dT_read
            end
            breathing_transport!(work, breathing_proj, breathing_work, grid, trial_dt, f0; camb=breathing_camb, Kz=Kz)
            deposition = apply_settling!(work, grid, trial_dt, sediment_params)
            bed_exchange!(work, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(work, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(virtual_oysters)
                oysters_backup = deepcopy(virtual_oysters)
                update_oysters!(work, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
                virtual_oysters = oysters_backup
            end
            state, work = work, state
            min_dt_taken = min(min_dt_taken, trial_dt); max_dt_taken = max(max_dt_taken, trial_dt)
            current_dt = dt_safe
        end

        if !breathing
        step_successful = false
        while !step_successful
            # Snapshot the committed state into the reusable trial buffer (no allocation),
            # then run the trial step in-place on `work`. A rejected step simply re-copies
            # from `state` on the next iteration; a successful step commits via an O(1) swap.
            _copy_dynamic_state!(work, state)
            oysters_backup = deepcopy(virtual_oysters)   # cheap (empty for kernel runs)

            apply_boundary_conditions!(work, grid, boundary_conditions)

            # Hydrodynamics Step. Backward runs read the real field at t_real = origin - lag and negate
            # the velocity (the adjoint of linear advection-diffusion); w is negated together with u,v so
            # a diagnosed omega stays consistent with the reversed horizontal field. Forward is untouched.
            hydro_time = reverse_time ? reverse_time_origin - (time + trial_dt) : time + trial_dt
            if ds !== nothing && hydro_data !== nothing
                update_hydrodynamics!(work, grid, ds, hydro_data, hydro_time; diagnose_w=diagnose_vertical_velocity)
            else
                update_hydrodynamics_placeholder!(work, grid, hydro_time)
            end
            if reverse_time
                @. work.u = -work.u; @. work.v = -work.v; @. work.w = -work.w
            end

            # Transport Step
            if advection_scheme == :ImplicitADI_3D
                for tracer_name in keys(work.tracers)
                    C_initial = work.tracers[tracer_name]
                    C_buffer1 = work._buffer1[tracer_name]
                    C_buffer2 = work._buffer2[tracer_name]

                    advect_diffuse_tvd_implicit_x!(C_buffer1, C_initial, work, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_y!(C_buffer2, C_buffer1, work, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_z!(C_initial, C_buffer2, work, grid, trial_dt, Kz, limiter_func)
                end
            else
                horizontal_transport!(work, grid, trial_dt, advection_scheme, D_crit, boundary_conditions; Kh=Kh)
                vertical_transport!(work, grid, trial_dt; Kz=Kz)
            end

            # --- Physics Steps ---
            deposition = apply_settling!(work, grid, trial_dt, sediment_params)
            bed_exchange!(work, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(work, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(work, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end

            # --- Timestep Validation ---
            # FFSL is stable at large advective Courant; its limit is the velocity-gradient
            # (Lipschitz) CFL instead, so the adaptive controller uses that term for :FFSL.
            cfl_term = advection_scheme == :FFSL ? calculate_max_gradient_cfl_term(work, grid) :
                                                   calculate_max_cfl_term(work, grid)
            cfl_actual = cfl_term * trial_dt

            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / (cfl_actual + 1e-9))
            else
                state, work = work, state    # O(1) commit; old `state` becomes the next trial buffer
                virtual_oysters = oysters_backup
                step_successful = true

                min_dt_taken = min(min_dt_taken, trial_dt)
                max_dt_taken = max(max_dt_taken, trial_dt)

                if use_adaptive_dt && cfl_actual < 0.5 * cfl_max
                    current_dt = min(dt_max, trial_dt * dt_growth_factor)
                else
                    current_dt = trial_dt
                end
            end
        end
        end  # if !breathing

        time += trial_dt
        state.time = time

        if !isempty(receptor_monitors) && time >= next_receptor_monitor_time - 1e-9
            for monitor in receptor_monitors
                write_receptor_monitor!(monitor, grid, state, time)
            end
            next_receptor_monitor_time += receptor_monitor_interval
        end

        if write_full_state && output_dir !== nothing && time >= next_full_state_output_time - 1e-9
            output_filename = joinpath(output_dir, "state_t_$(round(Int, time)).jld2")
            if tracer_only_output
                # Kernel product only: keep just the tracer fields (+ time). The dropped hydro and
                # scratch/flux buffers are reconstructable on the fly from the source .nc via
                # update_hydrodynamics!. Keeps the "state" key + `.tracers` so readers (e.g. E1) are
                # unchanged. NOTE: not a valid restart checkpoint (no hydro/buffers).
                jldsave(output_filename; state=(; tracers=state.tracers, time=state.time))
            else
                jldsave(output_filename; state=state, virtual_oysters=virtual_oysters)
            end
            next_full_state_output_time += full_state_output_interval
        end

        elapsed_wall_time_min = (time_ns() - start_wall_time) / 1e9 / 60
        ProgressMeter.update!(pbar, floor(Int, time - effective_start_time); showvalues = [
            (:sim_time_h, round(time / 3600, digits=1)),
            (:wall_time_m, round(elapsed_wall_time_min, digits=1)),
            (:current_timestep_s, round(trial_dt, digits=2)),
            (:min_timestep_s, round(min_dt_taken, digits=2)),
            (:max_timestep_s, round(max_dt_taken, digits=2))
        ])
    end
    ProgressMeter.finish!(pbar)

    # Flush all monitors at the end of the simulation
    for monitor in receptor_monitors
        flush_receptor_monitor!(monitor)
    end

    return state
end


"""
    run_and_store_simulation(grid, initial_state, sources, start_time, end_time, dt, output_interval; ...)

Runs a simulation and stores the state at specified intervals in memory.
...
"""
function run_and_store_simulation(grid::AbstractGrid, initial_state::State, sources::Vector{PointSource}, start_time::Float64, end_time::Float64, dt::Float64, output_interval::Float64;
                                  ds::Union{NCDataset, Nothing}=nothing,
                                  hydro_data::Union{HydrodynamicData, Nothing}=nothing,
                                  use_adaptive_dt::Bool=false,
                                  cfl_max::Float64=0.8,
                                  dt_max::Float64=dt,
                                  dt_min::Float64=0.1,
                                  dt_growth_factor::Float64=1.1,
                                  boundary_conditions::Vector{<:BoundaryCondition}=Vector{BoundaryCondition}(),
                                  functional_interactions::Vector{FunctionalInteraction}=Vector{FunctionalInteraction}(),
                                  sediment_params::Dict{Symbol, SedimentParams}=Dict{Symbol, SedimentParams}(),
                                  virtual_oysters::Vector{VirtualOyster}=Vector{VirtualOyster}(),
                                  oyster_tracers::NamedTuple=NamedTuple(),
                                  advection_scheme::Symbol=:FFSL,
                                  limiter_func::Function=FluxLimitersModule.van_leer, # <-- NEW ARGUMENT
                                  Kh::Float64=1.0,
                                  Kz::Float64=1e-4,
                                  D_crit::Float64=0.0,
                                  write_full_state::Bool=true,
                                  full_state_output_interval::Union{Float64, Nothing}=output_interval,
                                  receptor_monitors::Vector{ReceptorMonitor}=ReceptorMonitor[],
                                  receptor_monitor_interval::Union{Float64, Nothing}=output_interval)
                                  
    state = deepcopy(initial_state)
    time = start_time
    current_dt = dt
    results = [(state=deepcopy(state), oysters=deepcopy(virtual_oysters))]
    timesteps = [start_time]
    next_full_state_output_time = start_time + (full_state_output_interval !== nothing ? full_state_output_interval : output_interval)
    next_receptor_monitor_time = start_time + (receptor_monitor_interval !== nothing ? receptor_monitor_interval : output_interval)

    desc_str = (ds !== nothing) ? "Simulating & Storing (Real Data)..." : "Simulating & Storing (Test Mode)..."
    pbar = Progress(floor(Int, end_time - time); desc=desc_str, dt=1.0)
    
    while time < end_time
        trial_dt = use_adaptive_dt ? min(current_dt, dt_max) : dt

        dt_bound = end_time - time
        if write_full_state
            dt_bound = min(dt_bound, next_full_state_output_time - time)
        end
        if !isempty(receptor_monitors)
            dt_bound = min(dt_bound, next_receptor_monitor_time - time)
        end

        trial_dt = min(trial_dt, dt_bound)

        if use_adaptive_dt && trial_dt < dt_min
            @warn "\nWarning: Timestep below minimum threshold. Stopping simulation."
            break
        end
        if trial_dt < 1e-9; break; end

        step_successful = false
        while !step_successful
            state_backup = deepcopy(state)
            oysters_backup = deepcopy(virtual_oysters)

            apply_boundary_conditions!(state_backup, grid, boundary_conditions)
            
            if ds !== nothing && hydro_data !== nothing
                update_hydrodynamics!(state_backup, grid, ds, hydro_data, time + trial_dt)
            else
                # updates it with whatever is currently encoded in the placeholder 
                # probably a vortex
                update_hydrodynamics_placeholder!(state_backup, grid, time + trial_dt)
            end

            if advection_scheme == :ImplicitADI_3D
                # if we do advection/diffusion in one go we no longer need to call for them seperately
                for tracer_name in keys(state_backup.tracers)
                    C_initial = state_backup.tracers[tracer_name]
                    C_buffer1 = state_backup._buffer1[tracer_name]
                    C_buffer2 = state_backup._buffer2[tracer_name]
                    
                    # Call the new TVD functions
                    advect_diffuse_tvd_implicit_x!(C_buffer1, C_initial, state_backup, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_y!(C_buffer2, C_buffer1, state_backup, grid, trial_dt, Kh, limiter_func)
                    advect_diffuse_tvd_implicit_z!(C_initial, C_buffer2, state_backup, grid, trial_dt, Kz, limiter_func)
                end
            else
                horizontal_transport!(state_backup, grid, trial_dt, advection_scheme, D_crit, boundary_conditions; Kh=Kh)
                vertical_transport!(state_backup, grid, trial_dt; Kz=Kz)
            end
            
            deposition = apply_settling!(state_backup, grid, trial_dt, sediment_params)
            bed_exchange!(state_backup, grid, trial_dt, deposition, sediment_params)
            source_sink_terms!(state_backup, grid, sources, functional_interactions, time + trial_dt, trial_dt, D_crit)
            if !isempty(oysters_backup)
                update_oysters!(state_backup, grid, oysters_backup, trial_dt, oyster_tracers.dissolved, oyster_tracers.sorbed)
            end
            
            cfl_actual = calculate_max_cfl_term(state_backup, grid) * trial_dt
            
            if use_adaptive_dt && cfl_actual > cfl_max
                trial_dt = max(dt_min, trial_dt * 0.9 * cfl_max / (cfl_actual + 1e-9))
            else
                state = state_backup
                virtual_oysters = oysters_backup
                step_successful = true
                if use_adaptive_dt && cfl_actual < 0.5 * cfl_max
                    current_dt = min(dt_max, trial_dt * dt_growth_factor)
                else
                    current_dt = trial_dt
                end
            end
        end
        
        time += trial_dt
        state.time = time
        
        if !isempty(receptor_monitors) && time >= next_receptor_monitor_time - 1e-9
            for monitor in receptor_monitors
                write_receptor_monitor!(monitor, grid, state, time)
            end
            next_receptor_monitor_time += receptor_monitor_interval !== nothing ? receptor_monitor_interval : output_interval
        end

        if write_full_state && time >= next_full_state_output_time - 1e-9
            push!(results, (state=deepcopy(state), oysters=deepcopy(virtual_oysters)))
            push!(timesteps, time)
            next_full_state_output_time += full_state_output_interval !== nothing ? full_state_output_interval : output_interval
        end
        ProgressMeter.update!(pbar, floor(Int, time - start_time))
    end
    ProgressMeter.finish!(pbar)

    for monitor in receptor_monitors
        flush_receptor_monitor!(monitor)
    end

    return results, timesteps
end

end # module TimeSteppingModule