using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots

run_all_tests();



# init grid
g = initialize_grid(10, 10, 5, 100.0, 100.0, 20.0);

# init tracers
tracers = (:C_dissolved, :C_sorbed);

# init State
stat = initialize_state(g, tracers);

fin_state = run_simulation(g, stat, 0.0, 3.0, 1.0);

# lets try and visalize a tracer

begin
    # set up domain
    nx, ny, nz = 80, 40, 1 
    Lx, Ly, Lz = 1000.0, 500.0, 10.0
    grid = initialize_grid(nx, ny, nz, Lx, Ly, Lz);

    # Set up time 
    start_time = 0.0
    end_time = 600.0
    dt = 10.0

    # state
    tracer_names = (:C_dissolved, :C_sorbed);
    state = initialize_state(grid, tracer_names);
    state.tracers

    # setup fake data
    center_x = Lx / 4 # Start the blob on the left side
    center_y = Ly / 2
    width = Lx / 15

    peak_concentration = 1000.0
    background_concentration = 0.0
    min_concentration_threshold = 1e-6 # Anything below this is considered zero

    center_x = Lx / 4
    center_y = Ly / 2
    width = Lx / 15
    C = state.tracers[:C_dissolved];
    for k in 1:nz, j in 1:ny, i in 1:nx
        x = grid.x[i,j,k]
        y = grid.y[i,j,k]
        # Calculate the normalized Gaussian value (0 to 1)
        gaussian_value = exp(-((x - center_x)^2 / (2*width^2) + (y - center_y)^2 / (2*width^2)))
        if gaussian_value < min_concentration_threshold
            C[i,j,k] = background_concentration
        else
            C[i,j,k] = background_concentration + peak_concentration * gaussian_value
        end
    end


    # set up speed of water flow
    state.u .= 0.5;  # Constant velocity to the right (0.5 m/s)
    state.v .= 0.0;  # No velocity in y


    # inital mass 
    initial_mass = sum(C .* grid.volume);

    p_initial = heatmap(C[:,:,1]', title="Initial State (Time = 0s)", colormap=:viridis);
    final_state = run_simulation(grid, state, start_time, end_time, dt);
    final_C = final_state.tracers[:C_dissolved];
    final_mass = sum(final_C .* grid.volume);
    p_final = heatmap(final_C[:,:,1]', title="Final State (Time = $(end_time)s)", colormap=:viridis);
end


