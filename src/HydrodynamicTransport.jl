module HydrodynamicTransport

include("ModelStructs.jl")
include("GridModule.jl")
include("StateModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("SourceSinkModule.jl")
include("Hydrodynamics.jl")
include("TimeSteppingModule.jl")
include("TestCasesModule.jl") 


using .ModelStructs
export Grid, State, HydrodynamicData, PointSource

using .GridModule
export initialize_grid

using .StateModule
export initialize_state


using .HorizontalTransportModule
using .VerticalTransportModule
using .SourceSinkModule
using .HydrodynamicsModule

using .TimeSteppingModule
export run_simulation
export run_and_store_simulation

using .TestCasesModule
export run_all_tests

end # module HydrodynamicTransport