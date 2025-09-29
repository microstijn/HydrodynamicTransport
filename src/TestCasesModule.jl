# src/TestCases.jl

module TestCasesModule

export run_all_tests

using Test
using ..ModelStructs
using ..GridModule
using ..StateModule
using ..HorizontalTransportModule
using ..VerticalTransportModule
using ..SourceSinkModule
using ..VectorOperationsModule
using StaticArrays

# --- Helper function to set up a standard Cartesian grid and state ---
function setup_cartesian_test(;nx=20, ny=20, nz=5)
    Lx, Ly, Lz = 100.0, 100.0, 10.0
    grid = initialize_cartesian_grid(nx, ny, nz, Lx, Ly, Lz)
    state = initialize_state(grid, (:C,))
    return grid, state
end

# --- Helper function to set up a standard Curvilinear grid and state ---
function setup_curvilinear_test()
    nx, ny, nz = 20, 20, 5
    lon = [Float64(i) for i in 1:nx, j in 1:ny]; lat = [Float64(j) for i in 1:nx, j in 1:ny]
    z_w = collect(range(0, -10, length=nz+1))
    pm = ones(nx, ny); pn = ones(nx, ny); angle = zeros(nx,ny); h = ones(nx,ny) .* 10.0
    mask = trues(nx, ny)
    face_area_x = ones(nx+1, ny, nz); face_area_y = ones(nx, ny+1, nz); volume = ones(nx, ny, nz)
    
    grid = CurvilinearGrid(nx, ny, nz, lon, lat, lon, lat, lon, lat, z_w, 
                           pm, pn, angle, h, mask, mask, mask, 
                           face_area_x, face_area_y, volume)
    state = initialize_state(grid, (:C,))
    return grid, state
end


# --- Main Test Runner ---
function run_all_tests()
    @testset "All Unit Tests" begin
        
        @testset "1D Diffusion on a 3-cell Gradient" begin
            grid, state = setup_cartesian_test(nx=3, ny=1, nz=1)
            C = state.tracers[:C]
            C_in = similar(C)

            C[1,1,1] = 0.0; C[2,1,1] = 100.0; C[3,1,1] = 0.0
            copyto!(C_in, C)

            dt = 1.0; Kh = 1.0
            C_out = similar(C)
            HorizontalTransportModule.diffuse_x!(C_out, C_in, grid, dt, Kh)

            @test C_out[2,1,1] < 100.0
            @test C_out[1,1,1] > 0.0
            @test C_out[3,1,1] > 0.0
            mass_initial = sum(C_in .* grid.volume)
            mass_final = sum(C_out .* grid.volume)
            @test isapprox(mass_initial, mass_final, rtol=1e-12)
        end

        @testset "Grid and State Initialization" begin
            @testset "CartesianGrid initialization is correct" begin
                grid, _ = setup_cartesian_test()
                @test grid.dims == SVector(20, 20, 5)
                @test isapprox(grid.x[2,1,1] - grid.x[1,1,1], 5.0)
                @test isapprox(grid.y[1,2,1] - grid.y[1,1,1], 5.0)
            end

            @testset "State initialization from grid is correct" begin
                cart_grid, cart_state = setup_cartesian_test()
                @test size(cart_state.tracers[:C]) == (20, 20, 5)
                @test size(cart_state.u) == (21, 20, 5)
                @test size(cart_state.v) == (20, 21, 5)

                curv_grid, curv_state = setup_curvilinear_test()
                @test size(curv_state.tracers[:C]) == (20, 20, 5)
                @test size(curv_state.u) == (21, 20, 5)
                @test size(curv_state.v) == (20, 21, 5)
            end
        end

        @testset "Grid Operations" begin
            @testset "interpolate_center_to_xface! is correct" begin
                grid, state = setup_cartesian_test()
                for i in 1:grid.dims[1]; state.tracers[:C][i, :, :] .= i; end
                
                xface_C = zeros(size(state.u))
                GridModule.interpolate_center_to_xface!(xface_C, state.tracers[:C], grid)
                
                @test xface_C[10, 10, 3] == 9.5
            end

            @testset "rotate_velocities_to_geographic is correct" begin
                grid, state = setup_curvilinear_test()
                grid.angle .= π / 2.0 
                state.u .= 1.0
                
                u_east, v_north = rotate_velocities_to_geographic(grid, state.u, state.v)
                
                @test isapprox(u_east[10,10,3], 0.0, atol=1e-9)
                @test isapprox(v_north[10,10,3], 1.0, atol=1e-9)
            end
        end

        @testset "Physics Modules" begin
            @testset "Horizontal Advection is Monotonic and Conservative" begin
                grid, state = setup_curvilinear_test()
                C = state.tracers[:C]; C[8:10, 8:10, :] .= 100.0
                state.u .= 0.1; state.v .= 0.1
                dt = 1.0
                
                mass_initial = sum(C .* grid.volume)
                max_C_initial = maximum(C); min_C_initial = minimum(C)

                C_temp = similar(C)
                HorizontalTransportModule.advect_x!(C_temp, C, state.u, grid, dt)
                HorizontalTransportModule.advect_y!(C, C_temp, state.v, grid, dt)
                
                mass_final = sum(C .* grid.volume)
                @test isapprox(mass_final, mass_initial, rtol=1e-6)
                @test maximum(C) <= max_C_initial + 1e-9
                @test minimum(C) >= min_C_initial - 1e-9
            end

            @testset "TVD Advection Scheme: Solid Body Rotation Test" begin
                nx, ny = 50, 50
                grid, state = setup_cartesian_test(nx=nx, ny=ny, nz=1)
                C = state.tracers[:C]
                
                cone_radius = 15.0; cone_center_x = 50.0; cone_center_y = 75.0
                for j in 1:ny, i in 1:nx
                    dist = sqrt((grid.x[i,j,1] - cone_center_x)^2 + (grid.y[i,j,1] - cone_center_y)^2)
                    C[i,j,1] = max(0.0, 100.0 * (1.0 - dist / cone_radius))
                end
                
                period = 200.0; omega = 2π / period
                center_x = 50.0; center_y = 50.0
                u_centered = zeros(Float64, nx, ny, 1); v_centered = zeros(Float64, nx, ny, 1)
                for j in 1:ny, i in 1:nx
                    rx = grid.x[i,j,1] - center_x; ry = grid.y[i,j,1] - center_y
                    u_centered[i,j,1] = -omega * ry
                    v_centered[i,j,1] =  omega * rx
                end
                for i in 2:nx; state.u[i,:,:] = 0.5 * (u_centered[i-1,:,:] + u_centered[i,:,:]); end
                for j in 2:ny; state.v[:,j,:] = 0.5 * (v_centered[:,j-1,:] + v_centered[:,j,:]); end

                mass_initial = sum(C .* grid.volume)
                max_C_initial = maximum(C); min_C_initial = minimum(C)

                dt = 1.0; n_steps = Int(round(period / dt))
                C_temp = similar(C)
                for _ in 1:n_steps
                    HorizontalTransportModule.advect_x!(C_temp, C, state.u, grid, dt)
                    HorizontalTransportModule.advect_y!(C, C_temp, state.v, grid, dt)
                end

                mass_final = sum(C .* grid.volume)
                @test isapprox(mass_final, mass_initial, rtol=1e-6)
                @test maximum(C) <= max_C_initial + 1e-9
                @test minimum(C) >= min_C_initial - 1e-9
            end
            
            @testset "Horizontal Diffusion is Conservative and Reduces Peaks" begin
                grid, state = setup_curvilinear_test()
                C = state.tracers[:C]
                # --- FIX: Use a single-cell peak to avoid zero gradient at the center ---
                C[9, 9, :] .= 100.0
                dt = 0.1; Kh = 1.0
                
                mass_initial = sum(C .* grid.volume)
                max_C_initial = maximum(C)
                
                C_temp = similar(C)
                C_in = similar(C)
                copyto!(C_in, C)
                
                HorizontalTransportModule.diffuse_x!(C_temp, C_in, grid, dt, Kh)
                copyto!(C_in, C_temp)
                HorizontalTransportModule.diffuse_y!(C, C_in, grid, dt, Kh)

                mass_final = sum(C .* grid.volume)
                @test isapprox(mass_final, mass_initial, rtol=1e-6)
                @test maximum(C) < max_C_initial
            end

            @testset "Vertical Transport conserves mass" begin
                grid, state = setup_cartesian_test()
                C = state.tracers[:C]; C[:, :, 2:3] .= 100.0
                state.w[:, :, 3] .= 0.01
                dt = 1.0
                
                mass_initial = sum(C .* grid.volume)
                VerticalTransportModule.vertical_transport!(state, grid, dt)
                mass_final = sum(C .* grid.volume)

                @test isapprox(mass_final, mass_initial, rtol=1e-6)
            end
        end
        
        @testset "SourceSink Module" begin
            @testset "Point Source adds mass correctly" begin
                grid, state = setup_cartesian_test()
                source_rate = 10.0
                source_config = [PointSource(i=10, j=10, k=1, tracer_name=:C, influx_rate=(t)->source_rate)]
                dt = 60.0
                
                mass_initial = sum(state.tracers[:C] .* grid.volume)
                SourceSinkModule.source_sink_terms!(state, grid, source_config, 0.0, dt)
                mass_final = sum(state.tracers[:C] .* grid.volume)
                
                @test isapprox(mass_final - mass_initial, source_rate * dt, rtol=1e-9)
            end
        end

    end
end

end # module TestCasesModule