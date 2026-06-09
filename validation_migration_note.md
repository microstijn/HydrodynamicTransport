# Validation and Migration Note: Generic Receptor Monitoring

## 1. Migration / Usage Updates

### Old Usage
Previously, `run_simulation` and `run_and_store_simulation` only supported saving the full-grid 3D state using the `output_interval` parameter.
```julia
HydrodynamicTransport.run_simulation(
    grid, state, sources, start_time, end_time, dt;
    output_dir = "output",
    output_interval = 21600.0,  # Wrote full state every 6 hours
    # ... other args
)
```
Existing scripts structured this way will continue to work perfectly and the behavior remains identical: full states will be written every `output_interval`.

### New Usage
The simulation now supports **high-frequency, lightweight receptor monitoring** decoupled from full-state outputs. You can specify different output frequencies for full state vs receptor outputs, or entirely disable full state writing.

```julia
monitor_a = HydrodynamicTransport.create_receptor_monitor_from_lonlat(
    grid;
    receptor_id = "RECEPTOR_A",
    lon = -1.5, lat = 47.0,
    tracer_names = [:TRACER_1],
    normalization_by_tracer = Dict(:TRACER_1 => 1.0e12),
    output_file = "output/RECEPTOR_A_monitor.csv",
    start_time_seconds = start_time
)

HydrodynamicTransport.run_simulation(
    grid, state, sources, start_time, end_time, dt;
    output_dir = "output",
    # Old parameter acts as a fallback default, but you can be explicit:
    write_full_state = false,
    full_state_output_interval = 24.0 * 3600.0,

    receptor_monitors = [monitor_a],
    receptor_monitor_interval = 1.0 * 3600.0,  # CSV rows every 1 hr
)
```

## 2. Validation Requirements

You must run the following three tests manually in your environment to validate correctness.

### Test A: Backward Compatibility Test
**Objective**: Ensure existing simulations run exactly as before.
1. Run a script using only the `output_interval` and `output_dir` arguments.
2. Provide no monitor arguments (`receptor_monitors = ReceptorMonitor[]`).
3. **Pass Condition**: The full domain state files (`state_t_*.jld2`) are correctly saved at the specified `output_interval`.

### Test B: Receptor Monitor Smoke Test
**Objective**: Ensure the CSV monitors output the expected rows without writing full states.
1. Run a short test (e.g. 2 hours of simulated time) with `receptor_monitor_interval = 1.0 * 3600.0`.
2. Configure 2 receptors and 2 tracers.
3. Set `write_full_state = false`.
4. **Pass Condition**: No `state_t_*.jld2` files are generated. Two distinct CSV files are produced, each containing 4 rows (assuming the model writes at `start_time + interval` and end time) or exactly the mathematically predicted amount based on time boundaries. The CSV must contain `time_seconds`, `receptor_id`, `tracer_id`, `raw_value`, `clamped_value`, and `kernel_value`.

### Test C: Comparison Against Full-State Post-Processing
**Objective**: Guarantee that receptor monitor extraction matches full-state data physically.
1. Run a short simulation with `write_full_state = true` and both `full_state_output_interval` and `receptor_monitor_interval` set to exactly the same value (e.g. 1 hour).
2. Wait for the simulation to finish.
3. Use external post-processing tools to load the `state_t_*.jld2` files.
4. Calculate the column mean (if 3D) or cell value at the exact array index reported by the receptor CSV.
5. **Pass Condition**: The independently computed volume-weighted value must agree with the `raw_value` recorded in the receptor monitor CSV to machine numerical precision.
