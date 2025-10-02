using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots
using NCDatasets




#TODO
#=
Step 5: Refactor Helper Functions in HorizontalTransportModule.jl
Goal: Ensure that calculations of cell size (dx, dy) are correct for cells near the boundary.

Analysis: I have identified a critical bug. Functions like get_dx_at_face currently take i_phys and j_phys as arguments and use them to index into the pm and pn arrays. This is incorrect. The pm and pn arrays now have ghost cells and must be indexed with the full global indices (i_glob, j_glob) to correctly access the extrapolated metric values in the ghost region.

Proposed Change: I will change the signature of all grid-spacing helper functions (e.g., get_dx_at_face, get_dy_centers) to accept the global indices (i_glob, j_glob) instead of the physical ones. The main advection/diffusion loops will be updated to pass these correct indices. This is a crucial fix for stability.

Step 6: Implement VerticalTransportModule.jl for Curvilinear Grids
Goal: Add the necessary methods to perform vertical transport on a curvilinear grid.

Analysis: This module currently only has methods for CartesianGrid. We need to add new methods that can be dispatched for a CurvilinearGrid.

Proposed Change:

I will add a new method for solve_implicit_diffusion_column! that is specialized for CurvilinearGrid. It will correctly calculate dz from the grid.z_w vector and the vertical face area from 1 / (pm * pn).

I will update the main vertical_transport! function to correctly handle the CurvilinearGrid case, using its new helper.

Step 7: Create a New Integration Test
Goal: Add a final test to IntegrationTestsModule.jl that runs a short, end-to-end simulation on the curvilinear grid to prove that all the components work together.

Analysis: A simple test that confirms stability and mass conservation is the best way to validate the entire refactoring effort.

Proposed Change: I will add a new @testset "Curvilinear Simulation" that:

Initializes the grid and state from the ROMS file.

Sets all boundary velocities to zero to enforce a no-flux, closed-domain condition.

Initializes a simple tracer patch in the middle of the domain.

Runs the simulation for a few hundred steps.

Asserts that the total tracer mass is conserved and that no NaN values are produced.

This plan systematically addresses all the remaining pieces of the refactoring. Once this is complete, the model will be fully capable of running stable, conservative simulations on real-world curvilinear grids.
=#

# write tests to test the overall working of curvi and regular grids again
# then  to speed up the process. its too slow right now. 


@info "Setting up Real Data simulation..."
# OPeNDAP URL and variable map
norway_netcdf_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
norway_variable_map = Dict(
	:u => "u",
	:v => "v",
	:temp => "temp",
	:salt => "salt",
	:time => "ocean_time"
)
norway_hydro_data = HydrodynamicData(norway_netcdf_filepath, norway_variable_map)
# Simulation time and sources
norway_start_time = 0.0
norway_end_time = 2 * 3600.0
norway_dt = 120.0 * 12
norway_output_interval = 60*60*5.0
norway_source_config = [
	PointSource(i=100, j=300, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7),
	PointSource(i=200, j=300, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7),
	PointSource(i=300, j=300, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7),
    PointSource(i=10, j=20, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7),
	PointSource(i=80, j=90, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7),
	#PointSource(i=500, j=100, k=1, tracer_name=:C_virus, influx_rate=(t)->1.5e7)
]
# Connect to the remote dataset
@info "Connecting to remote NetCDF file via OPeNDAP..."
ds = NCDataset(norway_netcdf_filepath)
@info "Connection successful."
# --- UPDATED: Initialize grid and state correctly for curvilinear data ---
@info "Initializing Curvilinear Grid from NetCDF..."
norway_grid = initialize_curvilinear_grid(norway_hydro_data.filepath)

@info "Initializing State with NetCDF dimensions..."
norway_initial_state = initialize_state(norway_grid, ds, (:C_virus,))
# Run the simulation
@info "Starting simulation on real data..."
norway_f_state = run_simulation(
    norway_grid,
    norway_initial_state,
    norway_source_config,
    ds,
    norway_hydro_data,
    norway_start_time,
    norway_end_time,
    norway_dt
)

close(ds)
@info "Real data simulation finished."
