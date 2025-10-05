# HydrodynamicTransport.jl

`HydrodynamicTransport.jl` is a three-dimensional numerical model designed to simulate the fate and transport of dissolved or suspended substances in an aquatic environment. The model is architected to be "offline-coupled," meaning it is driven by pre-computed velocity and environmental data from sources like ROMS, rather than computing the hydrodynamics itself.

## Features

*   **Grid Support**: Works with both `CartesianGrid` and curvilinear `CurvilinearGrid` systems.
*   **Advection Schemes**: Implements multiple horizontal advection schemes, including a high-order, Total Variation Diminishing (TVD) scheme based on Bott (1989) and a simpler 3rd-Order Upstream (UP3) scheme.
*   **Stable Vertical Transport**: Uses an explicit first-order upwind scheme for vertical advection and a numerically stable implicit Crank-Nicolson scheme for vertical diffusion.
*   **Flexible Boundary Conditions**: Supports `OpenBoundary`, `RiverBoundary`, and `TidalBoundary` types to handle various inflow/outflow scenarios.
*   **Utilities**: Includes helper functions to initialize grids from NetCDF files, estimate stable timesteps (CFL condition), and automatically map variables from data files.

## Getting Started

### Prerequisites

*   [Julia](https://julialang.org/downloads/) (Version 1.6 or later recommended)

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

## Running the Tests

The package includes a test suite to verify its core functionality. To run the tests, activate the project environment and use the `test` command in the Julia package manager:

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
*   **Advection**: Implemented using either the Bott (1989) TVD scheme or a 3rd-order upstream scheme.
*   **Diffusion**: Solved with an explicit scheme.

#### Vertical Transport (`VerticalTransportModule.jl`)
*   **Advection**: Solved with an explicit, first-order upwind scheme.
*   **Diffusion**: Solved with a numerically stable implicit Crank-Nicolson scheme, which avoids the strict time step limitations of an explicit solver.

#### Sources & Sinks (`SourceSinkModule.jl`)
*   Flexibly handles point sources and includes a simple first-order decay model for specific tracers.

### Hydrodynamic Forcing

The model runs in an "offline-coupled" mode. The `Hydrodynamics.jl` module is responsible for updating the velocity and environmental fields at each time step, either from a placeholder analytical solution (for testing) or by reading and interpolating data from a NetCDF file.

## License

This project is licensed under the MIT License.