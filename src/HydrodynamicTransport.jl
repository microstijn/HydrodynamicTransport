# src/HydrodynamicTransport.jl

module HydrodynamicTransport

# --- 1. Include all source files to define the modules ---
include("ModelStructs.jl")
include("GridModule.jl")
include("ProjectionModule.jl")
include("StateModule.jl")
include("UtilsModule.jl")
include("FluxLimitersModule.jl")
include("VectorOperationsModule.jl")
include("BoundaryConditionsModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("BreathingTransportModule.jl")
include("SourceSinkModule.jl")
include("Hydrodynamics.jl")

# --- ADDED FOR SEDIMENTATION ---
include("SettlingModule.jl")
include("BedExchangeModule.jl")
# ---------------------------------
# Added oyster
include("OysterModule.jl")

include("ReceptorMonitoringModule.jl")
include("TimeSteppingModule.jl")



# Bring the contents of the modules into the main module's scope ---
using .ModelStructs
using .GridModule
using .ProjectionModule
using .StateModule
using .VectorOperationsModule
using .FluxLimitersModule
using .BoundaryConditionsModule
using .HorizontalTransportModule
using .VerticalTransportModule
using .SourceSinkModule
using .HydrodynamicsModule
using .OysterModule
using .ReceptorMonitoringModule
using .TimeSteppingModule
using .UtilsModule
# Note: The new modules are used internally by TimeSteppingModule,
# so they don't need to be brought into the main scope with 'using' here.


# Export the public API 
# Types from ModelStructs.jl
export AbstractGrid, CartesianGrid, CurvilinearGrid, State, HydrodynamicData, PointSource, 
       BoundaryCondition, OpenBoundary, RiverBoundary, TidalBoundary, FunctionalInteraction,
       SedimentParams, DecayParams, OysterParams, OysterState, VirtualOyster # Export the new struct
       
export ReceptorMonitor, create_receptor_monitor_from_lonlat, create_receptor_monitor_from_xy, flush_receptor_monitor!

# Functions from GridModule.jl
export initialize_cartesian_grid, initialize_curvilinear_grid, rebuild_metrics!

# Functions/types from ProjectionModule.jl (opt-in breathing-sigma continuity correction)
export BreathingProjector, build_projector, project!, padded_depth!, write_omega_velocity!

# Functions from StateModule.jl
export initialize_state

# Functions from VectorOperationsModule.jl
export rotate_velocities_to_geographic

# Functions from TimeSteppingModule.jl
export run_simulation, run_and_store_simulation

# Functions from UtilsModule.jl
export estimate_stable_timestep, create_hydrodynamic_data_from_file, lonlat_to_ij, calculate_max_cfl_term, calculate_max_gradient_cfl_term


end # module HydrodynamicTransport