# src/ReceptorMonitoringModule.jl

module ReceptorMonitoringModule

export ReceptorMonitor, create_receptor_monitor_from_lonlat, create_receptor_monitor_from_xy, flush_receptor_monitor!, write_receptor_monitor!

using ..HydrodynamicTransport.ModelStructs
using ..HydrodynamicTransport.UtilsModule: lonlat_to_ij

"""
    ReceptorMonitor

A generic mutable struct for monitoring tracer concentrations at specific receptor locations
without writing the full domain state.

# Fields
- `receptor_id::String`: Identifier for the receptor.
- `i::Int`, `j::Int`: Model/grid-facing receptor indices (for reporting).
- `i_array::Int`, `j_array::Int`: Actual array indices used to index tracer arrays (accounting for halos).
- `tracer_names::Vector{Symbol}`: Tracers to monitor.
- `normalization_by_tracer::Dict{Symbol,Float64}`: User-supplied normalization constants for each tracer.
- `output_file::String`: Path to the CSV output file.
- `start_time_seconds::Float64`: Simulation start time, used to report elapsed time.
- `extraction_mode::String`: Mode for extracting values (e.g., "volume_weighted_column_mean").
- `clamp_negative_values::Bool`: If true, `kernel_value = clamped_value / normalization_value`. If false, `kernel_value = raw_value / normalization_value`.
- `flush_every_n_monitor_times::Int`: Number of monitor times to buffer before flushing to disk.
- `buffer::Vector{NamedTuple}`: Internal buffer for rows before writing.
"""
Base.@kwdef mutable struct ReceptorMonitor
    receptor_id::String
    i::Int
    j::Int
    i_array::Int
    j_array::Int
    tracer_names::Vector{Symbol}
    normalization_by_tracer::Dict{Symbol,Float64}
    output_file::String
    start_time_seconds::Float64
    extraction_mode::String = "volume_weighted_column_mean"
    clamp_negative_values::Bool = true
    flush_every_n_monitor_times::Int = 24
    buffer::Vector{NamedTuple} = NamedTuple[]
end

"""
    get_tracer_array(tracers, tracer::Symbol)

Robustly fetches a tracer array from the tracers storage.
"""
function get_tracer_array(tracers, tracer::Symbol)
    if tracers isa AbstractDict
        haskey(tracers, tracer) && return tracers[tracer]
        haskey(tracers, string(tracer)) && return tracers[string(tracer)]
        return nothing
    else
        try
            hasproperty(tracers, tracer) && return getproperty(tracers, tracer)
        catch
        end
        return nothing
    end
end

"""
    receptor_vertical_weights(grid, i_array::Int, j_array::Int)

Calculates the vertical weights for volume-weighted column mean extraction.
For 3D grids with `volume`, returns normalized positive volumes.
Otherwise, returns uniform weights `1.0 / nz`.
"""
function receptor_vertical_weights(grid, i_array::Int, j_array::Int)
    nz = Int(grid.nz)

    if hasproperty(grid, :volume)
        # Assuming grid.volume has dimensions that can be indexed [i, j, k]
        vols = [Float64(grid.volume[i_array, j_array, k]) for k in 1:nz]
        posvols = [isfinite(v) && v > 0 ? v : 0.0 for v in vols]
        total = sum(posvols)

        if total > 0.0
            return posvols ./ total
        end
    end

    return fill(1.0 / nz, nz)
end

"""
    receptor_monitor_rows(monitor::ReceptorMonitor, grid, state, time_seconds::Float64)

Extracts tracer values for the receptor and returns a vector of rows (NamedTuples)
to be appended to the monitor's buffer.
"""
function receptor_monitor_rows(monitor::ReceptorMonitor, grid, state, time_seconds::Float64)
    rows = NamedTuple[]
    elapsed_seconds = time_seconds - monitor.start_time_seconds
    elapsed_hours = elapsed_seconds / 3600.0

    weights = receptor_vertical_weights(grid, monitor.i_array, monitor.j_array)
    nz = length(weights)

    for tracer in monitor.tracer_names
        A = get_tracer_array(state.tracers, tracer)
        if A === nothing
            @warn "Tracer '$tracer' not found in state.tracers for receptor '$(monitor.receptor_id)'. Skipping."
            continue
        end

        # Extract the column
        vals = [Float64(A[monitor.i_array, monitor.j_array, k]) for k in 1:nz]

        raw_value = sum(vals .* weights)
        clamped_value = sum(max.(vals, 0.0) .* weights)

        normalization_value = get(monitor.normalization_by_tracer, tracer, NaN)

        # Calculate kernel value as per user instructions
        if isfinite(normalization_value) && normalization_value > 0
            value_to_normalize = monitor.clamp_negative_values ? clamped_value : raw_value
            kernel_value = value_to_normalize / normalization_value
        else
            kernel_value = NaN
            @warn "Invalid receptor-monitor normalization value for tracer '$tracer' at receptor '$(monitor.receptor_id)': $normalization_value. Setting kernel_value to NaN."
        end

        local_min_raw = minimum(vals)
        local_max_raw = maximum(vals)
        local_negative_count = count(<(0.0), vals)
        local_negative_fraction = local_negative_count / nz

        row = (
            time_seconds = time_seconds,
            elapsed_seconds_since_start = elapsed_seconds,
            elapsed_hours_since_start = elapsed_hours,
            receptor_id = monitor.receptor_id,
            grid_i = monitor.i,
            grid_j = monitor.j,
            array_i = monitor.i_array,
            array_j = monitor.j_array,
            tracer_id = string(tracer),
            extraction_mode = monitor.extraction_mode,
            raw_value = raw_value,
            clamped_value = clamped_value,
            normalization_value = normalization_value,
            kernel_value = kernel_value,
            local_min_raw = local_min_raw,
            local_max_raw = local_max_raw,
            local_negative_count = local_negative_count,
            local_negative_fraction = local_negative_fraction
        )
        push!(rows, row)
    end

    return rows
end

"""
    write_receptor_monitor!(monitor::ReceptorMonitor, grid, state, time_seconds::Float64)

Buffers receptor monitor output for the current time and flushes if the threshold is reached.
"""
function write_receptor_monitor!(monitor::ReceptorMonitor, grid, state, time_seconds::Float64)
    rows = receptor_monitor_rows(monitor, grid, state, time_seconds)
    append!(monitor.buffer, rows)

    rows_per_time = length(monitor.tracer_names)
    threshold = monitor.flush_every_n_monitor_times * rows_per_time

    if length(monitor.buffer) >= threshold
        flush_receptor_monitor!(monitor)
    end
end

"""
    flush_receptor_monitor!(monitor::ReceptorMonitor)

Flushes the buffered rows to the CSV file using a lightweight writer.
"""
function flush_receptor_monitor!(monitor::ReceptorMonitor)
    isempty(monitor.buffer) && return

    out_dir = dirname(monitor.output_file)
    if !isempty(out_dir)
        mkpath(out_dir)
    end

    is_new_file = !isfile(monitor.output_file)

    # Lightweight CSV writing
    open(monitor.output_file, "a") do io
        if is_new_file
            # Write header
            header_keys = keys(first(monitor.buffer))
            write(io, join(string.(header_keys), ","))
            write(io, "\n")
        end

        for row in monitor.buffer
            row_values = map(v -> v isa String ? "\"$v\"" : string(v), values(row))
            write(io, join(row_values, ","))
            write(io, "\n")
        end
    end

    empty!(monitor.buffer)
end

"""
    create_receptor_monitor_from_lonlat(...)

Creates a ReceptorMonitor using geographic coordinates, leveraging `lonlat_to_ij`.
"""
function create_receptor_monitor_from_lonlat(
    grid;
    receptor_id::String,
    lon::Float64,
    lat::Float64,
    tracer_names::Vector{Symbol},
    normalization_by_tracer::Dict{Symbol,Float64},
    output_file::String,
    start_time_seconds::Float64,
    extraction_mode::String = "volume_weighted_column_mean"
)
    result = lonlat_to_ij(grid, lon, lat)
    result === nothing && error("Could not map receptor coordinates (lon: $lon, lat: $lat) to grid for receptor '$receptor_id'")

    i = Int(result[1])
    j = Int(result[2])

    halo = hasproperty(grid, :ng) ? Int(grid.ng) : 0

    return ReceptorMonitor(
        receptor_id = receptor_id,
        i = i,
        j = j,
        i_array = i + halo,
        j_array = j + halo,
        tracer_names = tracer_names,
        normalization_by_tracer = normalization_by_tracer,
        output_file = output_file,
        start_time_seconds = start_time_seconds,
        extraction_mode = extraction_mode
    )
end

"""
    create_receptor_monitor_from_xy(...)

Creates a ReceptorMonitor using projected (X,Y) coordinates.
Requires the grid to have an `xy_to_ij` method (or similar).
If not implemented by the package, this function must be adapted.
"""
function create_receptor_monitor_from_xy(
    grid;
    receptor_id::String,
    x::Float64,
    y::Float64,
    tracer_names::Vector{Symbol},
    normalization_by_tracer::Dict{Symbol,Float64},
    output_file::String,
    start_time_seconds::Float64,
    extraction_mode::String = "volume_weighted_column_mean"
)
    # If the package has an xy_to_ij, we'd use it here.
    # We will assume a hypothetical `xy_to_ij` exists or error gracefully.
    error("create_receptor_monitor_from_xy is currently unmapped; ensure xy_to_ij exists in UtilsModule to use this function.")

    # Example placeholder:
    # result = xy_to_ij(grid, x, y)
    # result === nothing && error("Could not map receptor coordinates to grid")
    # i = Int(result[1])
    # j = Int(result[2])
    # halo = hasproperty(grid, :ng) ? Int(grid.ng) : 0
    # return ReceptorMonitor(...)
end

end # module ReceptorMonitoringModule