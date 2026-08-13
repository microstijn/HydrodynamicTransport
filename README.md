# HydrodynamicTransport.jl

`HydrodynamicTransport.jl` is a three-dimensional numerical model designed to simulate the fate and transport of dissolved or suspended substances in an aquatic environment. The model is architected to be "offline-coupled," meaning it is driven by pre-computed velocity and environmental data from sources like ROMS, rather than computing the hydrodynamics itself.

## Features

*   **Grid Support**: Works with both `CartesianGrid` and curvilinear `CurvilinearGrid` systems.
*   **Advection Schemes**: Implements multiple horizontal advection schemes, including a high-order, Total Variation Diminishing (TVD) scheme based on Bott (1989), a simpler 3rd-Order Upstream (UP3) scheme, and an implicit ADI scheme (`:ImplicitADI_3D`).
*   **Stable Vertical Transport**: Uses an explicit first-order upwind scheme for vertical advection and a numerically stable implicit Crank-Nicolson scheme for vertical diffusion.
*   **Flexible Boundary Conditions**: Supports `OpenBoundary`, `RiverBoundary`, and `TidalBoundary` types to handle various inflow/outflow scenarios.
*   **Adaptive Time-Stepping**: Optional CFL-driven adaptive `dt` (`use_adaptive_dt=true`) with configurable bounds.
*   **Reactive Processes**: First-order tracer decay and arbitrary multi-tracer reactions via `FunctionalInteraction`, cohesive sediment settling/erosion (`SedimentParams`), and virtual filter-feeder uptake (`VirtualOyster`).
*   **Receptor Monitoring**: Write high-frequency point time series (`ReceptorMonitor`) without dumping the full grid state.
*   **Utilities**: Includes helper functions to initialize grids from NetCDF files, estimate stable timesteps (CFL condition), and automatically map variables from data files.

## Getting Started

### Prerequisites

*   [Julia](https://julialang.org/downloads/) 1.11 or later (developed and tested on 1.12).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd HydrodynamicTransport.jl
    ```

2.  **Enter the Julia REPL** by typing `julia` in your terminal.

3.  **Activate the project environment and instantiate dependencies:**
    ```julia
    julia> ]
    pkg> activate .
    pkg> instantiate
    ```
    This will install all the necessary packages listed in `Project.toml`.

## Basic Usage

Here is a simple example of setting up and running a simulation on a Cartesian grid.

```julia
using HydrodynamicTransport

# 1. Define grid dimensions and create the grid
nx, ny, nz = 20, 20, 5
Lx, Ly, Lz = 100.0, 100.0, 10.0
grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)

# 2. Initialize the model state with one tracer called :C
state = initialize_state(grid, (:C,))

# 3. Define a point source adding mass to the tracer
#    The influx rate is a function of time.
sources = [PointSource(i=10, j=10, k=1, tracer_name=:C, influx_rate=(t) -> 10.0)]

# 4. Define boundary conditions (e.g., open boundaries on East/West sides)
bcs = [OpenBoundary(side=:East), OpenBoundary(side=:West)]

# 5. Run the simulation for 1 hour (3600 seconds) with a 60-second timestep
start_time = 0.0
end_time = 3600.0
dt = 60.0

final_state = run_simulation(
    grid, state, sources, start_time, end_time, dt;
    boundary_conditions=bcs,
    advection_scheme=:TVD
)

println("Simulation complete. Final time: \$(final_state.time) seconds.")
```

## Working with Real Hydrodynamic Data

For real simulations the grid and the velocity/environmental forcing come from a NetCDF file
(e.g. a ROMS or MARS3D history file). The package can auto-detect the grid geometry and the
relevant variables:

```julia
using HydrodynamicTransport
using NCDatasets

filepath = "path/to/hydro_history.nc"

# Build the curvilinear grid and auto-map variables (u, v, temp, salt, time, ...).
grid       = initialize_curvilinear_grid(filepath)
hydro_data = create_hydrodynamic_data_from_file(filepath)

# Open the dataset and initialize the state for one tracer.
ds    = NCDataset(filepath)
state = initialize_state(grid, ds, (:Tracer,))

# Recommend a stable timestep from the CFL condition.
dt = estimate_stable_timestep(hydro_data; advection_scheme=:TVD)

# Place a source at a geographic location.
i, j    = lonlat_to_ij(grid, -1.55, 47.2)
sources = [PointSource(i=i, j=j, k=grid.nz, tracer_name=:Tracer, influx_rate=(t)->1.0e6)]

final_state = run_simulation(
    grid, state, sources, 0.0, 24*3600.0, dt;
    ds = ds, hydro_data = hydro_data,
    boundary_conditions = [OpenBoundary(side=:East), OpenBoundary(side=:West)],
    advection_scheme = :TVD,
)
close(ds)
```

See [`examples/run_loire_simulation_with_oysters.jl`](examples/run_loire_simulation_with_oysters.jl)
for a full real-data run combining sources, adsorption/desorption, sediment settling, decay,
and virtual oysters.

## Advanced Features

These are all passed as keyword arguments to `run_simulation` (see the source for the full list):

*   **Adaptive time-stepping**: `use_adaptive_dt=true` with `cfl_max`, `dt_max`, `dt_min`,
    and `dt_growth_factor` lets the solver grow/shrink `dt` to stay within the CFL limit.
*   **Reactions** (`functional_interactions`): supply a `Vector{FunctionalInteraction}`. Each
    interaction's function receives `(concentrations, environment, dt)` — where `environment`
    exposes `T`, `S`, `TSS`, `UVB`, and `depth` — and returns a `Dict` of per-tracer changes.
*   **Sediment** (`sediment_params`): a `Dict{Symbol, SedimentParams}` enables settling and
    bed erosion/deposition for the listed tracers (initialize the state with
    `sediment_tracers=[...]` so the bed-mass field exists).
*   **Virtual oysters** (`virtual_oysters`, `oyster_tracers`): place `VirtualOyster` filter
    feeders that remove dissolved/sorbed tracers from their cell.
*   **Receptor monitoring** (`receptor_monitors`, `receptor_monitor_interval`): write per-point
    CSV time series with `ReceptorMonitor` (or `create_receptor_monitor_from_lonlat`), optionally
    with `write_full_state=false` to skip full-grid output. See
    [`examples/example_receptor_monitor.jl`](examples/example_receptor_monitor.jl).
*   **Checkpoint / restart**: `output_dir` + `output_interval` write `.jld2` snapshots;
    `restart_from="state_t_….jld2"` resumes from one.

## Running the Tests

The package ships a self-contained test suite (synthetic in-memory grids and NetCDF fixtures —
no external data or network access required). Run it with the Julia package manager:

```julia
julia> ]
pkg> activate .
pkg> test
```

## Core Concepts

This section contains the detailed technical documentation from the original README.

### The Governing Equation

The model solves the 3D advection-dispersion-reaction equation for a scalar concentration, $C$:

```math
\frac{\partial C}{\partial t} + \frac{\partial (u_i C)}{\partial x_i} - \frac{\partial}{\partial x_i} \left( k_i \frac{\partial C}{\partial x_i} \right) = \text{Sources} - \text{Sinks} \quad (i=1,2,3)
```

Where:
*   $\frac{\partial C}{\partial t}$ **(Local Rate of Change):** The net change in concentration at a fixed point over time.
*   $\frac{\partial (u_i C)}{\partial x_i}$ **(Advection):** Transport of the substance due to the bulk fluid velocity, $\vec{u}$.
*   $\frac{\partial}{\partial x_i} \left( k_i \frac{\partial C}{\partial x_i} \right)$ **(Turbulent Diffusion):** Mixing and spreading of the substance.
*   **Sources - Sinks (Reactions):** All non-transport processes that add or remove the substance.

The model solves this equation using the **operator splitting** method, where each process (horizontal transport, vertical transport, sources/sinks) is solved sequentially within a single time step.

### The Computational Grid

The model uses a structured, staggered **Arakawa 'C' grid**. Scalar quantities (like concentration) are located at the cell center, while vector quantities (velocities) are located on the cell faces, normal to the direction of flow.

```
        +------- v -------+
        |                 |
        |       C         |
        u                 u
        |                 |
        |                 |
        +------- v -------+
```

### Numerical Implementation

#### Horizontal Transport (`HorizontalTransportModule.jl`)
*   **Advection**: four schemes are available via the `scheme` argument.
    *   `:FFSL` — conservative **flux-form semi-Lagrangian** (Lin and Rood, 1996) in directional
        splitting, with a piecewise-parabolic sub-grid reconstruction under the Colella–Woodward
        monotonicity constraint, flux-corrected against a donor-cell base (Zalesak, 1979).
        Second-order accurate, positive-definite and peak-preserving; stable at advective Courant
        numbers above 1. **This is the scheme used for the PREVIR kernel campaign**, and the one the
        benchmark table below is quoted from.
    *   `:TVD` — Bott (1989) total-variation-diminishing scheme.
    *   `:UP3` — 3rd-order upstream-biased scheme.
    *   `:ImplicitADI` — alternating-direction implicit solve.
*   **Diffusion**: Solved with an explicit scheme.

#### Vertical Transport (`VerticalTransportModule.jl`)
*   **Advection**: first-order upwind, solved **implicitly**, so it is unconditionally stable at
    vertical Courant numbers well above unity.
*   **Diffusion**: Solved with a numerically stable implicit Crank-Nicolson scheme, which avoids the strict time step limitations of an explicit solver.

#### Validation

`test/runtests.jl` runs the benchmark suite in `test/benchmarks/`, which validates each operator
against cases with known analytical solutions and writes the result tables deposited at the repository
root (`advection_validation_results.csv`, `diffusion_validation_results.csv`,
`vertical_validation_results.csv`). The advection table carries **all three explicit horizontal
schemes at three resolutions**, so `:FFSL` can be compared against `:TVD` and `:UP3` directly.
Headline results for `:FFSL`:

| test | metric | result |
|---|---|---|
| uniform translation | empirical order *p* | 2.01 (successive L2 ratios 4.03, 4.05) |
| solid-body rotation, 1 revolution | peak retention | 0.96, no negative values |
| rotation at Courant 3 | peak retention | 0.98 (stable) |
| Zalesak slotted cylinder | min / max | 0.0 / 0.998 (no spurious over/undershoot) |
| all advection tests | relative mass drift | 1e-9 to 5e-9 |

Two caveats the table cannot state: the monotonicity limiters reduce the formal third-order
reconstruction to second order at smooth extrema, and the conservation floor is set by
single-precision tracer storage rather than by the scheme.

**Rigid-lid caveat.** Cell volumes and face areas are held at the reference bathymetry and do not
breathe with the free surface, which enters only as a wet/dry gate. The diagnosed vertical velocity
therefore closes the discrete volume budget for the depth-integrated non-divergent flow (residual
~1e-16) but leaves a barotropic tidal column convergence uncompensated. A mass-consistent
("breathing") reformulation that restores discrete continuity is implemented in
`BreathingTransportModule.jl` and is opt-in.

#### Sources & Sinks (`SourceSinkModule.jl`)
*   Flexibly handles point sources and includes a simple first-order decay model for specific tracers.

### Hydrodynamic Forcing

The model runs in an "offline-coupled" mode. The `Hydrodynamics.jl` module is responsible for updating the velocity and environmental fields at each time step, either from a placeholder analytical solution (for testing) or by reading and interpolating data from a NetCDF file.

## License

This project is licensed under the MIT License.