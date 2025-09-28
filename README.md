# HydrodynamicTransport.jl: Model Documentation

**Version:** 1.0 (Post-Staggered Grid Refactor)
**Date:** 2025-09-26

## 1. Overview

`HydrodynamicTransport.jl` is a three-dimensional numerical model designed to simulate the fate and transport of dissolved or suspended substances in an aquatic environment. The model is architected to be "offline-coupled," meaning it does not compute hydrodynamics itself but is instead driven by pre-computed velocity and environmental data from another source, such as a hydrodynamic model's NetCDF output.

The numerical core is inspired by the transport schemes of the SAM-3D model, emphasizing mass conservation and numerical stability through the use of a staggered grid and advanced advection algorithms.

## 2. The Governing Equation

The model solves the 3D advection-dispersion-reaction equation for a scalar concentration, $C$:

```math
\frac{\partial C}{\partial t} + \frac{\partial (u_i C)}{\partial x_i} - \frac{\partial}{\partial x_i} \left( k_i \frac{\partial C}{\partial x_i} \right) = \text{Sources} - \text{Sinks} \quad (i=1,2,3)
```

Where:

* $\frac{\partial C}{\partial t}$ **(Local Rate of Change):** The net change in concentration at a fixed point over time. This is the primary variable the model solves for.

* $\frac{\partial (u_i C)}{\partial x_i}$ **(Advection):** Transport of the substance due to the bulk fluid velocity, $\vec{u}$.

* $\frac{\partial}{\partial x_i} \left( k_i \frac{\partial C}{\partial x_i} \right)$ **(Turbulent Diffusion):** Mixing and spreading of the substance due to turbulence, parameterized by the diffusivity coefficient, $k_i$. *(Note: This term is currently implemented for the vertical direction only)*.

* **Sources - Sinks (Reactions):** All non-transport processes that add or remove the substance, such as chemical decay.

### Solution Method: Operator Splitting

The model solves this equation using the **operator splitting**, or fractional-step, method. Within a single time step, $\Delta t$, the processes are solved sequentially. The `TimeSteppingModule.jl` implements this as follows:

1. **Update Hydrodynamics:** Environmental fields are updated.

2. **Horizontal Transport:** The change in concentration due to horizontal advection is calculated.

3. **Vertical Transport:** The change due to vertical advection and diffusion is calculated.

4. **Sources & Sinks:** The change due to reaction terms is calculated.

## 3. The Computational Grid

### Horizontal: Arakawa 'C' Grid

The model is built on a structured **Arakawa 'C' staggered grid**. This grid arrangement is specifically chosen for its  mass conservation properties. Scalar and vector variables are defined at different locations within a grid cell:

* **Scalar Quantities** (e.g., Concentration `C`, Temperature `T`, Salinity `S`) are located at the **cell center**.

* **Vector Quantities** (e.g., velocities `u`, `v`) are located on the **cell faces**, normal to the direction of flow.

This can be visualized in 2D as:

```
        +------- v -------+
        |                 |
        |       C         |
        u       T, S      u
        |                 |
        |                 |
        +------- v -------+

```

### Vertical: Z-Coordinate System

The vertical dimension is discretized using a **z-coordinate system**, where the domain is split into layers of fixed vertical thickness.

## 4. Numerical Implementation

### Horizontal Transport

Horizontal transport is solved in `HorizontalTransportModule.jl`.

* **Advection:** The model uses the **Bott (1989) advection scheme**. This is a high-order, positive-definite, and conservative flux-based method. It avoids generating unphysical negative concentrations by using a multi-step algorithm involving polynomial fitting and non-linear flux limiting at each face.

* **Diffusion:** Horizontal diffusion is not yet implemented.

### Vertical Transport

Vertical transport is solved in `VerticalTransportModule.jl` for each water column independently.

* **Advection:** Solved with an explicit **First-Order Upwind** scheme.

* **Diffusion:** Solved with a numerically stable **implicit scheme**. This method results in a tridiagonal system of linear equations for each water column, which is solved efficiently using `LinearAlgebra.Tridiagonal`.

### Sources & Sinks (Reactions)

The `SourceSinkModule.jl` implements the reaction model.

* **Logic:** It flexibly checks for the presence of specific tracers (`:C_dissolved`, `:C_sorbed`) and applies a simple first-order decay to each one found.

* **Equation:** The concentration is updated using an explicit Euler step:
  

```math
C_{new} = C_{old} \cdot (1 - k_{decay} \cdot \Delta t)
```

  where the decay rate, $k_{decay}$, is currently hard-coded to 10% per day.

## 5. Hydrodynamic Forcing

The model runs in an "offline-coupled" mode, driven by external data.

* **Implementation:** The `Hydrodynamics.jl` module is responsible for updating the velocity and environmental fields at each time step.

* **Current State:** It currently uses a **placeholder analytical solution** representing a reversing M2 tidal cycle, not real data. The velocities are calculated as:
  

```math
u(t) = 0.5 \cdot \cos\left(\frac{2 \pi t}{T}\right)
```

```math
v(t) = 0.2 \cdot \sin\left(\frac{2 \pi t}{T}\right)
```
  
  where $T$ is the M2 tidal period of 12.4 hours.
