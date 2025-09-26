module HydrodynamicTransport

# --- 1. Include all submodule files ---
include("ModelStructs.jl")
include("GridModule.jl")
include("StateModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("SourceSinkModule.jl")
include("Hydrodynamics.jl")
include("TimeSteppingModule.jl")
include("TestCasesModule.jl") 

# --- 2. Bring modules into scope and export the public API ---

using .ModelStructs
export Grid, State, HydrodynamicData

using .GridModule
export initialize_grid

using .StateModule
export initialize_state

# These are internal components called by the time stepper, no top-level export needed
using .HorizontalTransportModule
using .VerticalTransportModule
using .SourceSinkModule
using .HydrodynamicsModule

using .TimeSteppingModule
export run_simulation

using .TestCasesModule
export run_all_tests

end # module HydrodynamicTransport