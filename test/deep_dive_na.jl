using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using NCDatasets
using ProgressMeter
using LinearAlgebra

# Import all internal modules to make their functions available
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HydrodynamicsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.VerticalTransportModule: solve_implicit_diffusion_column!, _apply_sedimentation_backward_euler!


println("--- DEEP DIVE NaN DIAGNOSTIC SCRIPT (Updated) ---")

# --- 1. Simulation Setup (Copied from previr_SCRIPTING.jl) ---
loire_filepath = raw"D:\PreVir\loireModel\MARS3D\run_curviloire_2018.nc"
hydro_data = create_hydrodynamic_data_from_file(loire_filepath)
ds = NCDataset(loire_filepath)
grid = initialize_curvilinear_grid(loire_filepath)
tracer_names = (:Virus_Dissolved, :Virus_Sorbed,)
sediment_tracer_list = [:Virus_Sorbed]
state = initialize_state(grid, ds, tracer_names; sediment_tracers=sediment_tracer_list)
state.tss .= 5.0

sources = PointSource[]
sources_to_plot = [
    (name = "Nantes", lon = -1.549464, lat = 47.197319),
    (name = "Saint-Nazaire", lon = -2.28, lat = 47.27),
    (name = "Cordemais", lon = -1.97, lat = 47.28)
]
for s in sources_to_plot
    i, j = lonlat_to_ij(grid, s.lon, s.lat)
    push!(sources, PointSource(i=i, j=j, k=1, tracer_name=:Virus_Dissolved, influx_rate=(t)->1.0e10, relocate_if_dry=true))
end

sediment_params_dict = Dict(
    :Virus_Sorbed => SedimentParams(
        ws0 = 0.0005, tau_cr = 0.1, tau_d = 0.05, settling_scheme = :BackwardEuler
    )
)

function implicit_adsorption_desorption(concentrations, environment, dt)
    C_diss_old = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb_old = max(0.0, concentrations[:Virus_Sorbed])
    TSS = environment.TSS; Kd = 0.2; transfer_rate = 0.0001
    C_total = C_diss_old + C_sorb_old
    if C_total <= 0.0; return Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0); end
    alpha = dt * transfer_rate; beta = Kd * TSS
    numerator = C_sorb_old + alpha * beta * C_total
    denominator = 1.0 + alpha * (1.0 + beta)
    C_sorb_new = numerator / denominator
    delta_C = C_sorb_new - C_sorb_old

    if delta_C > 0; delta_C = min(delta_C, C_diss_old); else; delta_C = max(delta_C, -C_sorb_old); end
    
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end

virus_interaction = FunctionalInteraction(
    affected_tracers = [:Virus_Dissolved, :Virus_Sorbed],
    interaction_function = implicit_adsorption_desorption
)

start_time = 0.0
end_time = 48*60*60.0
dt = 300.0
advection_scheme = :ImplicitADI
D_crit = 0.05
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West), OpenBoundary(side=:North), OpenBoundary(side=:South)]

error_detected = false

# --- 2. Granular Instability Checking Function ---
function check_for_instability(var, var_name, tracer_name, step_name, time, i, j)
    global error_detected
    if error_detected; return true; end

    # --- UPDATED CHECK: Catch large positive values as well ---
    offending_index = findfirst(val -> isnan(val) || val < -10.0 || val > 1.0e20, var)

    if offending_index !== nothing
        error_detected = true
        offending_value = var[offending_index]
        println("\n" * "="^80)
        println("FATAL: Instability detected in variable '$var_name'")
        println("Tracer: '$tracer_name', Physics Step: '$step_name'")
        println("Time: $time, Grid Cell (i_glob=$i, j_glob=$j)")
        println("Offending Value: $offending_value at index $offending_index")
        
        if isa(offending_index, CartesianIndex)
            linear_idx = LinearIndices(var)[offending_index]
            println("Array Snippet (linearized): $(var[max(1, linear_idx-5):min(end, linear_idx+5)])")
        else
            println("Array Snippet: $(var[max(1, offending_index-5):min(end, offending_index+5)])")
        end
        
        println("="^80)
        return true
    end
    return false
end

# --- Generic Implicit Vertical Advection Solver ---
function solve_implicit_vertical_advection_column!(C_out_col, C_in_col, velocities, grid, i_glob, j_glob, dt)
    nz = length(C_in_col)
    if nz <= 1; C_out_col .= C_in_col; return; end

    a = Vector{Float64}(undef, nz-1); b = Vector{Float64}(undef, nz); c = Vector{Float64}(undef, nz-1)
    
    metric_product = grid.pm[i_glob, j_glob] * grid.pn[i_glob, j_glob]
    face_area = metric_product > 1e-12 ? 1 / metric_product : 0.0
    if face_area == 0.0; C_out_col .= C_in_col; return; end

    for k in 1:nz
        V_k = grid.volume[i_glob, j_glob, k]
        if V_k < 1e-9; b[k] = 1.0; if k > 1 a[k-1] = 0.0 end; if k < nz c[k] = 0.0 end; continue; end
        
        cr_k = (velocities[k] * face_area * dt) / V_k
        cr_kp1 = (velocities[k+1] * face_area * dt) / V_k
        
        if k > 1; a[k-1] = -max(cr_k, 0.0); end
        b[k] = 1.0 + max(cr_kp1, 0.0) - min(cr_k, 0.0)
        if k < nz; c[k] = min(cr_kp1, 0.0); end
    end
    
    A = Tridiagonal(a, b, c)
    C_out_col .= A \ C_in_col
end


# --- 3. Deep Dive Diagnostic Function ---
function run_diagnostic()
    
    # --- DEFINITIVE FIX: Local, corrected versions of implicit advection ---
    # These functions include the necessary wet/dry check.
    function advect_implicit_x_fixed!(C_intermediate, C_initial, state, grid, dt, D_crit)
        nx, ny, nz = (grid.nx, grid.ny, grid.nz); ng = grid.ng; u = state.u
        a = Vector{Float64}(undef, nx-1); b = Vector{Float64}(undef, nx); c = Vector{Float64}(undef, nx-1); d = Vector{Float64}(undef, nx)
        for j_phys in 1:ny, k in 1:nz
            j_glob = j_phys + ng
            is_row_wet = false
            for i_phys in 1:nx
                i_glob = i_phys + ng
                if grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, 1] > D_crit
                    is_row_wet = true; break
                end
            end
            if !is_row_wet
                view(C_intermediate, (ng+1):(nx+ng), j_glob, k) .= view(C_initial, (ng+1):(nx+ng), j_glob, k)
                continue
            end
            for i_phys in 1:nx
                i_glob = i_phys + ng
                u_left = (i_phys == 1) ? 0.0 : u[i_glob, j_glob, k]
                u_right = (i_phys == nx) ? 0.0 : u[i_glob+1, j_glob, k]
                dx_i = 1 / grid.pm[i_glob, j_glob]; dx_ip1 = 1 / grid.pm[i_glob+1, j_glob]
                cr_left = (dt / dx_i) * u_left; cr_right = (dt / dx_ip1) * u_right
                if i_phys > 1; a[i_phys-1] = -max(cr_left, 0.0); end
                b[i_phys] = 1.0 + max(cr_right, 0.0) - min(cr_left, 0.0)
                # --- FIX: Changed `gamma` to the correct variable ---
                if i_phys < nx; c[i_phys] = min(cr_right, 0.0); end
                d[i_phys] = C_initial[i_glob, j_glob, k]
            end
            A = Tridiagonal(a, b, c); solution = A \ d
            view(C_intermediate, (ng+1):(nx+ng), j_glob, k) .= solution
        end
    end

    function advect_implicit_y_fixed!(C_final, C_intermediate, state, grid, dt, D_crit)
        nx, ny, nz = (grid.nx, grid.ny, grid.nz); ng = grid.ng; v = state.v
        a = Vector{Float64}(undef, ny-1); b = Vector{Float64}(undef, ny); c = Vector{Float64}(undef, ny-1); d = Vector{Float64}(undef, ny)
        for i_phys in 1:nx, k in 1:nz
            i_glob = i_phys + ng
            is_col_wet = false
            for j_phys in 1:ny
                j_glob = j_phys + ng
                if grid.h[i_glob, j_glob] + state.zeta[i_glob, j_glob, 1] > D_crit
                    is_col_wet = true; break
                end
            end
            if !is_col_wet
                view(C_final, i_glob, (ng+1):(ny+ng), k) .= view(C_intermediate, i_glob, (ng+1):(ny+ng), k)
                continue
            end
            for j_phys in 1:ny
                j_glob = j_phys + ng
                v_bottom = (j_phys == 1) ? 0.0 : v[i_glob, j_glob, k]
                v_top = (j_phys == ny) ? 0.0 : v[i_glob, j_glob+1, k]
                dy_j = 1 / grid.pn[i_glob, j_glob]; dy_jp1 = 1 / grid.pn[i_glob, j_glob+1]
                cr_bottom = (dt / dy_j) * v_bottom; cr_top = (dt / dy_jp1) * v_top
                if j_phys > 1; a[j_phys-1] = -max(cr_bottom, 0.0); end
                b[j_phys] = 1.0 + max(cr_top, 0.0) - min(cr_bottom, 0.0)
                if j_phys < ny; c[j_phys] = min(cr_top, 0.0); end
                d[j_phys] = C_intermediate[i_glob, j_glob, k]
            end
            A = Tridiagonal(a, b, c); solution = A \ d
            view(C_final, i_glob, (ng+1):(ny+ng), k) .= solution
        end
    end
    
    println("\n--- STARTING DEEP DIVE DIAGNOSTIC ---")
    time_range = start_time:dt:end_time
    @showprogress "Simulating..." for time in time_range
        if time == start_time; continue; end
        state.time = time

        if error_detected; @goto end_sim; end

        # --- Step 1: Boundaries & Hydrodynamics ---
        apply_boundary_conditions!(state, grid, bcs)
        update_hydrodynamics!(state, grid, ds, hydro_data, time)
        
        # --- Step 2: Horizontal Transport (using fixed local versions) ---
        for tracer_name in keys(state.tracers)
            C_initial = state.tracers[tracer_name]
            C_intermediate = state._buffers[tracer_name]
            advect_implicit_x_fixed!(C_intermediate, C_initial, state, grid, dt, D_crit)
            advect_implicit_y_fixed!(C_initial, C_intermediate, state, grid, dt, D_crit)
        end
        for (name, arr) in state.tracers; arr .= max.(0.0, arr); end
        if check_for_instability(state.tracers[:Virus_Sorbed], "Tracer", :Virus_Sorbed, "Post Horizontal Transport", time, 0, 0); @goto end_sim; end


        # --- Step 3: Vertical Transport (Operator Splitting) ---
        Kz = 1e-4; g = 9.81; ng = grid.ng; nx, ny, nz = grid.nx, grid.ny, grid.nz
        for (tracer_name, C_initial) in state.tracers
            C_advected = state._buffers[tracer_name]
            C_settled = similar(C_initial)
            for j_phys in 1:ny, i_phys in 1:nx
                i_glob, j_glob = i_phys+ng, j_phys+ng
                C_in_col=view(C_initial,i_glob,j_glob,:); C_out_col=view(C_advected,i_glob,j_glob,:); w_faces=view(state.w,i_glob,j_glob,:)
                solve_implicit_vertical_advection_column!(C_out_col, C_in_col, w_faces, grid, i_glob, j_glob, dt)
            end
            C_advected.=max.(0.0, C_advected)
            if check_for_instability(C_advected,"C_advected",tracer_name,"Vertical Advection",time,0,0); @goto end_sim; end
            is_sediment = haskey(sediment_params_dict, tracer_name)
            if is_sediment
                params=sediment_params_dict[tracer_name]; ws_velocities=zeros(nz+1)
                for j_phys in 1:ny, i_phys in 1:nx
                    i_glob, j_glob = i_phys+ng, j_phys+ng
                    C_in_col=view(C_advected,i_glob,j_glob,:); C_out_col=view(C_settled,i_glob,j_glob,:)
                    for k in 2:nz; C_donor=C_in_col[k-1]; Cv=C_donor/params.rho_particle; ws_effective=params.ws0*(1.0-min(1.0,Cv))^params.n_exponent; ws_velocities[k]=-ws_effective; end
                    solve_implicit_vertical_advection_column!(C_out_col, C_in_col, ws_velocities, grid, i_glob, j_glob, dt)
                end
                C_settled.=max.(0.0,C_settled)
            else; C_settled.=C_advected; end
            if check_for_instability(C_settled,"C_settled",tracer_name,"Vertical Settling",time,0,0); @goto end_sim; end
            for j_phys in 1:ny, i_phys in 1:nx
                i_glob, j_glob = i_phys+ng, j_phys+ng
                C_in_col=view(C_settled,i_glob,j_glob,:); C_out_col=view(C_initial,i_glob,j_glob,:)
                solve_implicit_diffusion_column!(C_out_col, C_in_col, grid, i_glob, j_glob, dt, Kz)
            end
            C_initial.=max.(0.0, C_initial)
        end

        if error_detected; @goto end_sim; end
        
        # --- Step 4: Source/Sink & Bed Exchange ---
        source_sink_terms!(state, grid, sources, [virus_interaction], time, dt, D_crit)
        for (tracer_name, params) in sediment_params_dict
            bed_mass_array = state.bed_mass[tracer_name]; C_final_tracer = state.tracers[tracer_name]
            for j_phys in 1:ny, i_phys in 1:nx
                i_glob,j_glob = i_phys+ng, j_phys+ng
                C_col_out=view(C_final_tracer, i_glob, j_glob, :)
                _apply_sedimentation_backward_euler!(C_col_out, bed_mass_array, grid, state, params, i_glob, j_glob, dt, g, D_crit)
            end
        end

        for (name,arr) in state.tracers; arr.=max.(0.0,arr); if check_for_instability(arr,"tracer: $name",name,"End of Timestep",time,0,0); @goto end_sim; end; end
    end

    @label end_sim
    return
end

# --- 5. Run the diagnostic and clean up ---
run_diagnostic()

close(ds)
println("\n--- DIAGNOSTIC RUN COMPLETE ---")
println("Stopped at simulation time: $(state.time) seconds.")

