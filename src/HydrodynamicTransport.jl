module HydrodynamicTransport

include("ModelStructs.jl")
include("GridModule.jl")
include("StateModule.jl")
include("HorizontalTransportModule.jl")
include("VerticalTransportModule.jl")
include("TimeSteppingModule.jl")
include("TestCasesModule.jl")

using .ModelStructs
export Grid
export State

using .GridModule
export initialize_grid

using .StateModule
export initialize_state

using .HorizontalTransportModule
export horizontal_transport!

using .VerticalTransportModule
export vertical_transport!

using .TimeSteppingModule
export run_simulation

using .TestCasesModule 
export run_all_tests  

end # module HydrodynamicTransport
