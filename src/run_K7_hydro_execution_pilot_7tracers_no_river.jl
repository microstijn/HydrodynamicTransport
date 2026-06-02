#!/usr/bin/env julia

using Pkg
try
    Pkg.activate(joinpath(@__DIR__, ".."))
catch
end

using CSV
using DataFrames
using Dates
using Statistics
using Printf

# =============================================================================
# run_K7_hydro_execution_pilot_7tracers_no_river.jl
# =============================================================================
#
# Purpose
# -------
# Technical pilot runner for one K7 hydrodynamic transport execution using the
# 7 pilot source tracers and explicitly excluding RIVER_MAIN.
#
# Pilot source groups
# -------------------
# Included:
#   WWTP_TOUGAS
#   WWTP_STNAZAIRE_E
#   WWTP_STNAZAIRE_W
#   OVERFLOW_NEARFIELD
#   OVERFLOW_UPSTREAM
#   OVERFLOW_DOWNSTREAM
#   OVERFLOW_MISC
#
# Excluded:
#   RIVER_MAIN
#
# Key modelling decision: vertical release
# ----------------------------------------
# For each horizontal source member, release is distributed over the wet vertical
# column at that source cell, not over all wet cells in the estuary.
#
# Rationale:
#   - the source location is known horizontally but not vertically;
#   - distributing over the full estuary would create an artificial basin-wide
#     initial condition rather than a local source;
#   - distributing over all wet vertical layers of the source column avoids an
#     arbitrary surface/bottom-layer assumption;
#   - total tracer mass for each K7 source group is conserved.
#
# Mass convention:
#   total_mass_per_tracer_group = 1.0 arbitrary unit
#   group_member_weight values sum to 1 within each source group
#   vertical weights sum to 1 within each wet source column
#
# Therefore:
#   sum over all source members and wet layers of released mass = 1.0 per tracer
#
# Practical implementation:
#   Each horizontal source member receives:
#       member_mass = total_mass_per_tracer_group * group_member_weight
#
#   Each wet vertical layer receives:
#       cell_mass = member_mass * vertical_weight
#
#   For a 1-hour pulse:
#       influx_rate = cell_mass / release_duration_seconds
#
# The preferred vertical weighting is layer-thickness/volume weighting, giving a
# depth-uniform concentration in the source column. If layer thickness is not
# available from the model API, equal weighting over wet layers is allowed as a
# technical-pilot fallback, but it should be marked in the logs.
#
# IMPORTANT ADAPTER NOTE
# ----------------------
# HydrodynamicTransport / CurviLoire project APIs differ between working trees.
# This script has a clearly marked ADAPTER SECTION. If your local API already
# provides functions such as lonlat_to_ij, PointSource, and run_simulation, you
# usually only need to fill the adapter functions near the bottom.
#
# The script performs all manifest/geometry validation and writes a source release
# plan before calling the model. Set DRY_RUN_ONLY=true to validate without running
# the model.
#
# =============================================================================

# =============================================================================
# 1. CONFIG
# =============================================================================

PROJECT_ROOT = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT"
ANALYSIS_ROOT = joinpath(PROJECT_ROOT, "04_analysis")
MANIFEST_DIR = joinpath(ANALYSIS_ROOT, "transport_kernels", "K7", "manifests")
EXECUTION_ROOT = joinpath(ANALYSIS_ROOT, "transport_kernels", "K7", "executions")
HYDRO_EXECUTION_MANIFEST_FILE = joinpath(MANIFEST_DIR, "K7_hydro_execution_manifest_v4.csv")
SOURCE_GEOMETRY_FILE = joinpath(MANIFEST_DIR, "K7_source_geometry_v3_pilot_7tracers_no_river.csv")
RECEPTOR_GEOMETRY_FILE = joinpath(MANIFEST_DIR, "K7_receptor_geometry_v3_pilot_7tracers_no_river.csv")
PILOT_SOURCE_SET_FILE = joinpath(MANIFEST_DIR, "K7_pilot_7tracer_source_set_v3.csv")

# The v4 execution_id still says ALL8TRACERS because the execution manifest is
# the general 8-tracer campaign manifest. This pilot runner intentionally uses
# only the 7 included tracers from PILOT_SOURCE_SET_FILE.
PILOT_EXECUTION_ID_FROM_MANIFEST = "K7_S1_W1_ALL8TRACERS"
PILOT_RUN_ID = replace(PILOT_EXECUTION_ID_FROM_MANIFEST, "ALL8TRACERS" => "7TRACERS_NO_RIVER")
# Set this to false after the ADAPTER SECTION is wired to your local model API.
DRY_RUN_ONLY = true

# Conservative unit-pulse kernel settings.
TOTAL_MASS_PER_TRACER_GROUP = 1.0e12
NORMALIZE_OUTPUTS_TO_UNIT_RELEASE = true
KERNEL_NORMALIZATION_MASS = TOTAL_MASS_PER_TRACER_GROUP


# Vertical source distribution.
VERTICAL_RELEASE_MODE = "depth_weighted_wet_column"
ALLOW_EQUAL_VERTICAL_WEIGHTS_FALLBACK = true

# If true, source columns where no wet vertical layers are found are skipped and
# logged instead of hard-erroring. For a pilot, false is safer.
SKIP_SOURCE_IF_NO_WET_LAYERS = false

# =============================================================================
# 2. Utilities
# =============================================================================

function canonical_name(s::AbstractString)
    s = replace(s, '\ufeff' => "")
    s = strip(lowercase(s))
    s = replace(s, r"[^a-z0-9]+" => "_")
    s = replace(s, r"_+" => "_")
    s = replace(s, r"^_" => "")
    s = replace(s, r"_$" => "")
    return s
end

function normalize_names!(df::DataFrame)
    old_names = names(df)
    new_names = String[]
    used = Dict{String,Int}()
    for nm in old_names
        base = canonical_name(nm)
        if haskey(used, base)
            used[base] += 1
            base = base * "_" * string(used[base])
        else
            used[base] = 1
        end
        push!(new_names, base)
    end
    rename!(df, Pair.(old_names, new_names))
    return df
end

function require_cols(df::DataFrame, cols::Vector{Symbol}; label="DataFrame")
    missing_cols = [c for c in cols if !(string(c) in names(df))]
    isempty(missing_cols) || error("$(label) missing required columns: " * join(string.(missing_cols), ", "))
end

function parse_bool_any(x)
    if x isa Bool
        return x
    end
    s = lowercase(strip(string(x)))
    return s in ["true", "1", "yes", "y"]
end

function parse_float_any(x)
    if ismissing(x)
        return missing
    end
    if x isa Number
        return Float64(x)
    end
    s = strip(string(x))
    isempty(s) && return missing
    s = replace(s, "," => ".")
    return parse(Float64, s)
end

function safe_symbol(s)
    return Symbol(string(s))
end

function pulse_rate_function(release_start_seconds::Float64,
                             release_duration_seconds::Float64,
                             cell_mass::Float64)
    rate = cell_mass / release_duration_seconds
    release_end_seconds = release_start_seconds + release_duration_seconds
    return t -> ((release_start_seconds <= t < release_end_seconds) ? rate : 0.0)
end

function normalize_weights_by_group!(df::DataFrame)
    for g in groupby(df, :kernel_source_id)
        s = sum(skipmissing(g.group_member_weight))
        if !isfinite(s) || abs(s - 1.0) > 1e-6
            @warn "group_member_weight does not sum to 1; normalizing for pilot source group" first(g.kernel_source_id) s
            idx = parentindices(g)[1]
            vals = df[idx, :group_member_weight]
            vals2 = [ismissing(v) || !isfinite(Float64(v)) ? 0.0 : Float64(v) for v in vals]
            s2 = sum(vals2)
            if s2 <= 0
                df[idx, :group_member_weight] .= 1.0 / length(idx)
            else
                df[idx, :group_member_weight] .= vals2 ./ s2
            end
        end
    end
end

# =============================================================================
# 3. Load and validate inputs
# =============================================================================

println("============================================================")
println("K7 pilot runner: 7 tracers, no RIVER_MAIN")
println("============================================================")

for f in [HYDRO_EXECUTION_MANIFEST_FILE, SOURCE_GEOMETRY_FILE, RECEPTOR_GEOMETRY_FILE, PILOT_SOURCE_SET_FILE]
    isfile(f) || error("Missing required file: $(f)")
end

execution_manifest = CSV.read(HYDRO_EXECUTION_MANIFEST_FILE, DataFrame)
source_geometry = CSV.read(SOURCE_GEOMETRY_FILE, DataFrame)
receptor_geometry = CSV.read(RECEPTOR_GEOMETRY_FILE, DataFrame)
pilot_source_set = CSV.read(PILOT_SOURCE_SET_FILE, DataFrame)

normalize_names!(execution_manifest)
normalize_names!(source_geometry)
normalize_names!(receptor_geometry)
normalize_names!(pilot_source_set)

require_cols(execution_manifest,
    [:execution_id, :hydro_input_file, :simulation_start_time_seconds,
     :simulation_end_time_seconds, :tracer_release_time_seconds,
     :release_duration_hours, :output_interval_seconds, :use_adaptive_dt,
     :dt_initial_seconds, :cfl_max, :dt_max_seconds, :dt_min_seconds,
     :dt_growth_factor, :advection_scheme, :d_crit],
    label="execution_manifest")

require_cols(source_geometry,
    [:kernel_source_id, :source_id, :tracer_id, :geometry_role, :pilot_include,
     :lon, :lat, :group_member_weight, :relocate_if_dry, :geometry_status],
    label="source_geometry")

require_cols(receptor_geometry,
    [:kernel_receptor_id, :lon, :lat, :horizontal_extraction, :vertical_extraction,
     :geometry_status],
    label="receptor_geometry")

require_cols(pilot_source_set,
    [:kernel_source_id, :tracer_id, :pilot_include, :reason],
    label="pilot_source_set")

# Select execution row.
ex = execution_manifest[execution_manifest.execution_id .== PILOT_EXECUTION_ID_FROM_MANIFEST, :]
nrow(ex) == 1 || error("Expected one execution row for $(PILOT_EXECUTION_ID_FROM_MANIFEST), found $(nrow(ex))")
exrow = first(eachrow(ex))

# Coerce source flags.
source_geometry[!, :pilot_include] = parse_bool_any.(source_geometry.pilot_include)
pilot_source_set[!, :pilot_include] = parse_bool_any.(pilot_source_set.pilot_include)
source_geometry[!, :group_member_weight] = parse_float_any.(source_geometry.group_member_weight)
source_geometry[!, :lon] = parse_float_any.(source_geometry.lon)
source_geometry[!, :lat] = parse_float_any.(source_geometry.lat)
source_geometry[!, :relocate_if_dry] = parse_bool_any.(source_geometry.relocate_if_dry)

included_sources = pilot_source_set[pilot_source_set.pilot_include .== true, :]
excluded_sources = pilot_source_set[pilot_source_set.pilot_include .== false, :]

included_source_ids = Set(string.(included_sources.kernel_source_id))
excluded_source_ids = Set(string.(excluded_sources.kernel_source_id))

if "RIVER_MAIN" in included_source_ids
    error("RIVER_MAIN is included, but this runner is for the 7-tracer no-river pilot.")
end

pilot_geom = source_geometry[
    (source_geometry.pilot_include .== true) .&
    in.(source_geometry.kernel_source_id, Ref(collect(included_source_ids))) .&
    (source_geometry.geometry_status .== "usable"), :]

if any(pilot_geom.kernel_source_id .== "RIVER_MAIN")
    error("RIVER_MAIN unexpectedly present in pilot source geometry.")
end

missing_groups = setdiff(included_source_ids, Set(string.(unique(pilot_geom.kernel_source_id))))
isempty(missing_groups) || error("Pilot source geometry missing included groups: " * join(collect(missing_groups), ";"))

bad_rows = source_geometry[
    (source_geometry.pilot_include .== true) .&
    (source_geometry.geometry_status .!= "usable"), :]
if nrow(bad_rows) > 0
    @warn "Some pilot_include rows are not usable and were excluded from source instantiation." bad_rows[:, [:kernel_source_id, :source_id, :geometry_status]]
end

normalize_weights_by_group!(pilot_geom)

# Runtime parameters from manifest.
hydro_input_file = string(exrow.hydro_input_file)
simulation_start_time_seconds = Float64(exrow.simulation_start_time_seconds)
simulation_end_time_seconds = Float64(exrow.simulation_end_time_seconds)
tracer_release_time_seconds = Float64(exrow.tracer_release_time_seconds)
release_duration_seconds = Float64(exrow.release_duration_hours) * 3600.0
if !isfinite(release_duration_seconds) || release_duration_seconds <= 0
    release_duration_seconds = RELEASE_DURATION_SECONDS_DEFAULT
end
output_interval_seconds = Float64(exrow.output_interval_seconds)

# Pilot output directory.
base_output_directory = string(exrow.output_directory)
pilot_output_directory = replace(base_output_directory, "ALL8TRACERS" => "7TRACERS_NO_RIVER")
mkpath(pilot_output_directory)

source_release_plan_file = joinpath(pilot_output_directory, PILOT_RUN_ID * "_source_release_plan.csv")
run_metadata_file = joinpath(pilot_output_directory, PILOT_RUN_ID * "_run_metadata.csv")

println()
println("Pilot execution selected:")
println("  manifest execution_id = ", PILOT_EXECUTION_ID_FROM_MANIFEST)
println("  pilot run id          = ", PILOT_RUN_ID)
println("  hydro file            = ", hydro_input_file)
println("  start time seconds    = ", simulation_start_time_seconds)
println("  release time seconds  = ", tracer_release_time_seconds)
println("  end time seconds      = ", simulation_end_time_seconds)
println("  output interval       = ", output_interval_seconds)
println("  source geometry rows  = ", nrow(pilot_geom))
println("  source groups         = ", join(sort(collect(unique(pilot_geom.kernel_source_id))), ";"))
println("  dry run only          = ", DRY_RUN_ONLY)

# =============================================================================
# 4. ADAPTER SECTION
# =============================================================================
# Fill these functions with the actual local HydrodynamicTransport / CurviLoire
# API calls. The rest of this file should not need major edits.
# =============================================================================

function ADAPTER_load_model_context(hydro_input_file::AbstractString)
    # Import required modules
    @eval using HydrodynamicTransport
    @eval using NCDatasets

    # Initialize the hydrodynamic data and grid using the exact APIs found in the package
    hydro_data = HydrodynamicTransport.create_hydrodynamic_data_from_file(hydro_input_file)
    grid = HydrodynamicTransport.initialize_curvilinear_grid(hydro_input_file)
    ds = NCDataset(hydro_input_file)

    return (; grid=grid, hydro_data=hydro_data, ds=ds)
end

function ADAPTER_lonlat_to_ij(ctx, lon::Float64, lat::Float64)
    result = HydrodynamicTransport.lonlat_to_ij(ctx.grid, lon, lat)
    if result === nothing
        error("lonlat_to_ij returned nothing for lon=$lon, lat=$lat. Coordinates might be outside the grid.")
    end
    return result[1], result[2]
end

function ADAPTER_wet_vertical_layers_with_thickness(ctx, i::Int, j::Int, time_seconds::Float64)
    grid = ctx.grid
    ng = grid.ng
    i_glob = i + ng
    j_glob = j + ng

    # Check if the column is wet at all
    if !grid.mask_rho[i_glob, j_glob]
        return []
    end

    layers = []
    dz = grid.z_w[2:end] .- grid.z_w[1:end-1]

    for k in 1:grid.nz
        # For sigma coordinates, all levels in a wet column are considered wet
        push!(layers, (k=k, thickness=abs(dz[k])))
    end

    return layers
end

function ADAPTER_make_point_source(i::Int, j::Int, k::Int, tracer_name::Symbol, influx_rate; relocate_if_dry::Bool=true)
    return HydrodynamicTransport.PointSource(
        i = i,
        j = j,
        k = k,
        tracer_name = tracer_name,
        influx_rate = influx_rate,
        relocate_if_dry = relocate_if_dry
    )
end

function ADAPTER_run_simulation(ctx; sources, tracers, output_directory, start_time, end_time,
        output_interval_seconds, use_adaptive_dt, dt_initial_seconds, cfl_max,
        dt_max_seconds, dt_min_seconds, dt_growth_factor, advection_scheme, d_crit)

    # Initialize the state object
    tracer_names_tuple = Tuple(tracers)
    state = HydrodynamicTransport.initialize_state(ctx.grid, ctx.ds, tracer_names_tuple)

    # Empty physics params since we want a conservative tracer pilot
    sediment_params = Dict{Symbol, HydrodynamicTransport.SedimentParams}()
    virtual_oysters = HydrodynamicTransport.VirtualOyster[]
    functional_interactions = HydrodynamicTransport.FunctionalInteraction[]

    HydrodynamicTransport.run_simulation(
        ctx.grid,
        state,
        sources,
        start_time,
        end_time,
        dt_initial_seconds;
        ds = ctx.ds,
        hydro_data = ctx.hydro_data,
        use_adaptive_dt = use_adaptive_dt,
        cfl_max = cfl_max,
        dt_max = dt_max_seconds,
        dt_min = dt_min_seconds,
        dt_growth_factor = dt_growth_factor,
        sediment_params = sediment_params,
        virtual_oysters = virtual_oysters,
        functional_interactions = functional_interactions,
        advection_scheme = Symbol(advection_scheme),
        D_crit = d_crit,
        output_dir = output_directory,
        output_interval = output_interval_seconds,
        boundary_conditions = HydrodynamicTransport.BoundaryCondition[]
    )
end

# =============================================================================
# 5. Build source release plan and, if enabled, PointSource objects
# =============================================================================

function vertical_weights(ctx, i::Int, j::Int, t::Float64)
    layers = ADAPTER_wet_vertical_layers_with_thickness(ctx, i, j, t)
    if length(layers) == 0
        if SKIP_SOURCE_IF_NO_WET_LAYERS
            return DataFrame(k=Int[], vertical_weight=Float64[], layer_thickness=Float64[])
        else
            error("No wet vertical layers found for source column i=$(i), j=$(j)")
        end
    end

    ks = Int[]
    th = Float64[]
    for layer in layers
        push!(ks, Int(layer.k))
        val = Float64(layer.thickness)
        push!(th, isfinite(val) && val > 0 ? val : 0.0)
    end

    s = sum(th)
    if s <= 0
        if ALLOW_EQUAL_VERTICAL_WEIGHTS_FALLBACK
            @warn "Layer thickness unavailable/nonpositive; using equal vertical weights" i j
            th .= 1.0
            s = sum(th)
        else
            error("Cannot compute vertical weights for source column i=$(i), j=$(j): no positive layer thicknesses.")
        end
    end

    return DataFrame(k=ks, vertical_weight=th ./ s, layer_thickness=th)
end

function build_sources_and_plan(ctx, pilot_geom::DataFrame)
    source_objects = Any[]
    plan_rows = DataFrame[]

    for row in eachrow(pilot_geom)
        ksid = string(row.kernel_source_id)
        tracer_id = string(row.tracer_id)
        tracer_symbol = safe_symbol(tracer_id)
        lon = Float64(row.lon)
        lat = Float64(row.lat)
        member_weight = Float64(row.group_member_weight)
        member_mass = TOTAL_MASS_PER_TRACER_GROUP * member_weight
        relocate_if_dry = parse_bool_any(row.relocate_if_dry)

        i, j = ADAPTER_lonlat_to_ij(ctx, lon, lat)
        i = Int(i)
        j = Int(j)

        vw = vertical_weights(ctx, i, j, tracer_release_time_seconds)
        if nrow(vw) == 0
            @warn "Skipping source member with no wet vertical layers" ksid row.source_id i j
            continue
        end

        for vrow in eachrow(vw)
            cell_mass = member_mass * Float64(vrow.vertical_weight)
            influx = pulse_rate_function(tracer_release_time_seconds, release_duration_seconds, cell_mass)
            ps = ADAPTER_make_point_source(i, j, Int(vrow.k), tracer_symbol, influx; relocate_if_dry=relocate_if_dry)
            push!(source_objects, ps)

            push!(plan_rows, DataFrame(
                pilot_run_id = [PILOT_RUN_ID],
                kernel_source_id = [ksid],
                source_id = [string(row.source_id)],
                tracer_id = [tracer_id],
                lon = [lon],
                lat = [lat],
                grid_i = [i],
                grid_j = [j],
                grid_k = [Int(vrow.k)],
                group_member_weight = [member_weight],
                vertical_weight = [Float64(vrow.vertical_weight)],
                layer_thickness = [Float64(vrow.layer_thickness)],
                total_mass_per_tracer_group = [TOTAL_MASS_PER_TRACER_GROUP],
                member_mass = [member_mass],
                cell_mass = [cell_mass],
                release_start_seconds = [tracer_release_time_seconds],
                release_duration_seconds = [release_duration_seconds],
                release_rate = [cell_mass / release_duration_seconds],
                vertical_release_mode = [VERTICAL_RELEASE_MODE],
                relocate_if_dry = [relocate_if_dry]
            ))
        end
    end

    release_plan = isempty(plan_rows) ? DataFrame() : vcat(plan_rows..., cols=:union)
    return source_objects, release_plan
end

# =============================================================================
# 6. Dry-run plan or actual model run
# =============================================================================

# In dry run we cannot call adapter functions. Instead write a horizontal-only
# release plan and stop before model API calls.
if DRY_RUN_ONLY
    horizontal_plan = copy(pilot_geom[:, [
        :kernel_source_id,
        :source_id,
        :source_name,
        :source_type,
        :tracer_id,
        :geometry_role,
        :lon,
        :lat,
        :group_member_weight,
        :relocate_if_dry,
        :geometry_status
    ]])
    horizontal_plan[!, :pilot_run_id] .= PILOT_RUN_ID
    horizontal_plan[!, :total_mass_per_tracer_group] .= TOTAL_MASS_PER_TRACER_GROUP
    horizontal_plan[!, :member_mass] = TOTAL_MASS_PER_TRACER_GROUP .* horizontal_plan.group_member_weight
    horizontal_plan[!, :release_start_seconds] .= tracer_release_time_seconds
    horizontal_plan[!, :release_duration_seconds] .= release_duration_seconds
    horizontal_plan[!, :vertical_release_mode] .= VERTICAL_RELEASE_MODE
    horizontal_plan[!, :note] .= "Dry-run horizontal plan only; vertical split happens after grid wet layers are available."

    CSV.write(source_release_plan_file, horizontal_plan)

    metadata = DataFrame(
        key = [
            "pilot_run_id", "manifest_execution_id", "hydro_input_file",
            "simulation_start_time_seconds", "tracer_release_time_seconds",
            "simulation_end_time_seconds", "release_duration_seconds",
            "output_interval_seconds", "n_tracers", "n_horizontal_source_members",
            "river_main_excluded", "dry_run_only", "TOTAL_MASS_PER_TRACER_GROUP",
            "KERNEL_NORMALIZATION_MASS"
        ],
        value = string.([
            PILOT_RUN_ID, PILOT_EXECUTION_ID_FROM_MANIFEST, hydro_input_file,
            simulation_start_time_seconds, tracer_release_time_seconds,
            simulation_end_time_seconds, release_duration_seconds,
            output_interval_seconds, length(unique(pilot_geom.tracer_id)), nrow(pilot_geom),
            true, DRY_RUN_ONLY, TOTAL_MASS_PER_TRACER_GROUP,
            KERNEL_NORMALIZATION_MASS
        ])
    )
    CSV.write(run_metadata_file, metadata)

    println()
    println("DRY_RUN_ONLY=true, so no HydrodynamicTransport model call was made.")
    println("Wrote horizontal source release plan:")
    println("  ", source_release_plan_file)
    println("Wrote run metadata:")
    println("  ", run_metadata_file)
    println()
    println("Set DRY_RUN_ONLY=false and implement the ADAPTER SECTION to execute the model.")
    println("============================================================")
else
    ctx = ADAPTER_load_model_context(hydro_input_file)
    source_objects, release_plan = build_sources_and_plan(ctx, pilot_geom)

    if nrow(release_plan) == 0
        error("No source objects / release plan rows were generated.")
    end

    CSV.write(source_release_plan_file, release_plan)

    # Check mass conservation at the source-plan level.
    mass_by_tracer = combine(groupby(release_plan, :tracer_id), :cell_mass => sum => :released_mass)
    println()
    println("Mass by tracer in source release plan:")
    show(mass_by_tracer, allrows=true, allcols=true)
    println()

    for row in eachrow(mass_by_tracer)
        if abs(Float64(row.released_mass) - TOTAL_MASS_PER_TRACER_GROUP) > 1e-6
            @warn "Released mass differs from intended unit mass" row.tracer_id row.released_mass
        end
    end

    tracers = [safe_symbol(t) for t in sort(unique(pilot_geom.tracer_id))]

    metadata = DataFrame(
        key = [
            "pilot_run_id", "manifest_execution_id", "hydro_input_file",
            "simulation_start_time_seconds", "tracer_release_time_seconds",
            "simulation_end_time_seconds", "release_duration_seconds",
            "output_interval_seconds", "n_tracers", "n_horizontal_source_members",
            "n_point_sources_after_vertical_split", "river_main_excluded", "dry_run_only",
            "TOTAL_MASS_PER_TRACER_GROUP", "KERNEL_NORMALIZATION_MASS"
        ],
        value = string.([
            PILOT_RUN_ID, PILOT_EXECUTION_ID_FROM_MANIFEST, hydro_input_file,
            simulation_start_time_seconds, tracer_release_time_seconds,
            simulation_end_time_seconds, release_duration_seconds,
            output_interval_seconds, length(tracers), nrow(pilot_geom),
            length(source_objects), true, DRY_RUN_ONLY,
            TOTAL_MASS_PER_TRACER_GROUP, KERNEL_NORMALIZATION_MASS
        ])
    )
    CSV.write(run_metadata_file, metadata)

    println()
    println("Starting HydrodynamicTransport pilot run...")
    println("  n tracers             = ", length(tracers))
    println("  horizontal members    = ", nrow(pilot_geom))
    println("  vertical point sources= ", length(source_objects))
    println("  output directory      = ", pilot_output_directory)

    ADAPTER_run_simulation(ctx;
        sources = source_objects,
        tracers = tracers,
        output_directory = pilot_output_directory,
        start_time = simulation_start_time_seconds,
        end_time = simulation_end_time_seconds,
        output_interval_seconds = output_interval_seconds,
        use_adaptive_dt = parse_bool_any(exrow.use_adaptive_dt),
        dt_initial_seconds = Float64(exrow.dt_initial_seconds),
        cfl_max = Float64(exrow.cfl_max),
        dt_max_seconds = Float64(exrow.dt_max_seconds),
        dt_min_seconds = Float64(exrow.dt_min_seconds),
        dt_growth_factor = Float64(exrow.dt_growth_factor),
        advection_scheme = string(exrow.advection_scheme),
        d_crit = Float64(exrow.d_crit)
    )

    println()
    println("Pilot run complete.")
    println("============================================================")
end