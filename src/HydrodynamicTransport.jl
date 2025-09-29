# src/HydrodynamicTransport.jl

module HydrodynamicTransport

include("ModelStructs.jl")
include("GridModule.jl")
include("VectorOperationsModule.jl")
include("StateModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("SourceSinkModule.jl")
include("Hydrodynamics.jl")
include("TimeSteppingModule.jl")
include("TestCasesModule.jl")
include("IntegrationTestsModule.jl")

# Abstract and Concrete Grid type
using .ModelStructs
export AbstractGrid
export CartesianGrid
export CurvilinearGrid
export State
export HydrodynamicData
export PointSource

# grid initializers and interpolators
using .GridModule
export initialize_cartesian_grid
export initialize_curvilinear_grid
export interpolate_center_to_xface!
export interpolate_center_to_yface!

using .VectorOperationsModule
export rotate_velocities_to_geographic

using .StateModule
export initialize_state

# Internal physics 
using .HorizontalTransportModule
using .VerticalTransportModule
using .SourceSinkModule
using .HydrodynamicsModule

# TimeSteppingModule 
using .TimeSteppingModule
export run_simulation, run_and_store_simulation

# The test suite 
using .TestCasesModule
export run_all_tests

using .IntegrationTestsModule
export run_integration_tests

end # module HydrodynamicTransport