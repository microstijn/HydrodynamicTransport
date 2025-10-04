# src/HydrodynamicTransport.jl

module HydrodynamicTransport

# --- 1. Include all source files to define the modules ---
# Core modules are included first.
include("ModelStructs.jl")
include("GridModule.jl")
include("StateModule.jl")
include("VectorOperationsModule.jl")
include("BoundaryConditionsModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("SourceSinkModule.jl")
include("Hydrodynamics.jl")
include("TimeSteppingModule.jl")
include("UtilsModule.jl") # <-- MOVED UP to be with other core modules

# Test-related modules are included last.
#include("TestCasesModule.jl")
#include("IntegrationTestsModule.jl")


# --- 2. Bring the contents of the modules into the main module's scope ---
using .ModelStructs
using .GridModule
using .StateModule
using .VectorOperationsModule
using .BoundaryConditionsModule
using .HorizontalTransportModule
using .VerticalTransportModule
using .SourceSinkModule
using .HydrodynamicsModule
using .TimeSteppingModule
#using .TestCasesModule
#using .IntegrationTestsModule
using .UtilsModule


# --- 3. Export the public API ---
# Types from ModelStructs.jl
export AbstractGrid, CartesianGrid, CurvilinearGrid, State, HydrodynamicData, PointSource, 
       BoundaryCondition, OpenBoundary, RiverBoundary, TidalBoundary
       
# Functions from GridModule.jl
export initialize_cartesian_grid, initialize_curvilinear_grid

# Functions from StateModule.jl
export initialize_state

# Functions from VectorOperationsModule.jl
export rotate_velocities_to_geographic

# Functions from TimeSteppingModule.jl
export run_simulation, run_and_store_simulation

# Functions from TestCasesModule.jl and IntegrationTestsModule.jl
#export run_all_tests, run_integration_tests

# Functions from UtilsModule.jl
export estimate_stable_timestep
export create_hydrodynamic_data_from_file
export lonlat_to_ij

end # module HydrodynamicTransport