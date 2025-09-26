module HydrodynamicTransport

include("ModelStructs.jl")
include("GridModule.jl")
include("StateModule.jl")
include("HorizontalTransportModule.jl")
include("TimeSteppingModule.jl")


using .ModelStructs

export Grid
export State

using .GridModule
export initialize_grid

using .StateModule
export initialize_state

using .HorizontalTransportModule
export horizontal_transport!

using .TimeSteppingModule
export run_simulation




end # module HydrodynamicTransport
