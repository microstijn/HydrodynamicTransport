# test/new_implicit_tests.jl

# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Revise
using HydrodynamicTransport
using NCDatasets
using Statistics

# --- Bring all necessary functions into scope for testing ---
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.VectorOperationsModule
using HydrodynamicTransport.BoundaryConditionsModule
using HydrodynamicTransport.HorizontalTransportModule
using HydrodynamicTransport.VerticalTransportModule
using HydrodynamicTransport.SourceSinkModule
using HydrodynamicTransport.HydrodynamicsModule
using HydrodynamicTransport.TimeSteppingModule
using HydrodynamicTransport.UtilsModule

using HydrodynamicTransport.HorizontalTransportModule: advect_diffuse_implicit_x!, advect_diffuse_implicit_y!

# ==============================================================================
# --- TESTSET 6: 3D Implicit ADI Solver Components (Curvilinear Grid) ---
# ==============================================================================
@testset "6. 3D Implicit ADI Solver Components (Curvilinear Grid)" begin
    @info "Running Testset 6: 3D Implicit ADI Solver Components on a Curvilinear Grid..."

    # --- Setup a common grid and state for the tests ---
    ng = 2
    nx, ny, nz = 20, 20, 10
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    dx, dy, dz = 10.0, 10.0, 10.0

    pm = ones(nx_tot, ny_tot) ./ dx
    pn = ones(nx_tot, ny_tot) ./ dy
    h = ones(nx_tot, ny_tot) .* (nz * dz)
    z_w = collect(range(-(nz * dz), 0, length=nz+1))
    volume = fill(dx * dy * dz, (nx_tot, ny_tot, nz))
    face_area_x = fill(dy * dz, (nx_tot + 1, ny_tot, nz))
    face_area_y = fill(dx * dz, (nx_tot, ny_tot + 1, nz))
    
    grid = CurvilinearGrid(ng, nx, ny, nz, 
                           zeros(nx_tot,ny_tot), zeros(nx_tot,ny_tot),
                           zeros(nx_tot+1,ny_tot), zeros(nx_tot+1,ny_tot),
                           zeros(nx_tot,ny_tot+1), zeros(nx_tot,ny_tot+1),
                           z_w, pm, pn, zeros(nx_tot,ny_tot), h,
                           trues(nx_tot,ny_tot), trues(nx_tot+1,ny_tot), trues(nx_tot,ny_tot+1),
                           face_area_x, face_area_y, volume)

    state = initialize_state(grid, (:Tracer,))
    dt = 5.0
    Kh, Kz = 0.1, 0.01

    center_i, center_j, center_k = nx/2, ny/2, nz/2
    for k in 1:nz, j in 1:ny, i in 1:nx
        i_glob, j_glob = i + ng, j + ng
        dist_sq = (i - center_i)^2 + (j - center_j)^2 + (k - center_k)^2
        state.tracers[:Tracer][i_glob, j_glob, k] = 100.0 * exp(-dist_sq / 8)
    end
    
    initial_mass = sum(state.tracers[:Tracer][ng+1:nx+ng, ng+1:ny+ng, :] .* view(grid.volume, ng+1:nx+ng, ng+1:ny+ng, :))
    
    # --- Function to calculate center of mass for a more robust test ---
    function center_of_mass(C, grid)
        mass_x = 0.0; mass_y = 0.0; mass_z = 0.0; total_mass = 0.0
        for k in 1:grid.nz, j in 1:grid.ny, i in 1:grid.nx
            i_g, j_g = i+grid.ng, j+grid.ng
            m = C[i_g, j_g, k] * grid.volume[i_g, j_g, k]
            mass_x += m * i; mass_y += m * j; mass_z += m * k
            total_mass += m
        end
        return (mass_x/total_mass, mass_y/total_mass, mass_z/total_mass)
    end

    @testset "X-Sweep Advection and Diffusion" begin
        state_x = deepcopy(state)
        state_x.u .= 0.5
        # FIX: Set boundary velocities to zero for a closed box
        state_x.u[ng+1, :, :] .= 0.0
        state_x.u[nx+ng+1, :, :] .= 0.0
        
        C_in = deepcopy(state_x.tracers[:Tracer])
        C_out = state_x._buffer1[:Tracer]
        initial_com = center_of_mass(C_in, grid)

        advect_diffuse_implicit_x!(C_out, C_in, state_x, grid, dt, Kh)
        
        final_mass = sum(C_out[ng+1:nx+ng, ng+1:ny+ng, :] .* view(grid.volume, ng+1:nx+ng, ng+1:ny+ng, :))
        @test isapprox(final_mass, initial_mass, rtol=1e-7)

        final_com = center_of_mass(C_out, grid)
        @test final_com[1] > initial_com[1] # Robust check for movement
    end

    @testset "Y-Sweep Advection and Diffusion" begin
        state_y = deepcopy(state)
        state_y.v .= 0.5
        # FIX: Set boundary velocities to zero
        state_y.v[:, ng+1, :] .= 0.0
        state_y.v[:, ny+ng+1, :] .= 0.0

        C_in = deepcopy(state_y.tracers[:Tracer])
        C_out = state_y._buffer1[:Tracer]
        initial_com = center_of_mass(C_in, grid)

        advect_diffuse_implicit_y!(C_out, C_in, state_y, grid, dt, Kh)

        final_mass = sum(C_out[ng+1:nx+ng, ng+1:ny+ng, :] .* view(grid.volume, ng+1:nx+ng, ng+1:ny+ng, :))
        @test isapprox(final_mass, initial_mass, rtol=1e-7)

        final_com = center_of_mass(C_out, grid)
        @test final_com[2] > initial_com[2]
    end

    @testset "Z-Sweep Advection and Diffusion" begin
        state_z = deepcopy(state)
        state_z.w .= 0.1
        # FIX: Set boundary velocities to zero
        state_z.w[:, :, 1] .= 0.0 # Bottom face
        state_z.w[:, :, nz+1] .= 0.0 # Top face
        
        C_in = deepcopy(state_z.tracers[:Tracer])
        C_out = state_z._buffer1[:Tracer]
        initial_com = center_of_mass(C_in, grid)

        advect_diffuse_implicit_z!(C_out, C_in, state_z, grid, dt, Kz)
        
        final_mass = sum(C_out[ng+1:nx+ng, ng+1:ny+ng, :] .* view(grid.volume, ng+1:nx+ng, ng+1:ny+ng, :))
        @test isapprox(final_mass, initial_mass, rtol=1e-7)

        final_com = center_of_mass(C_out, grid)
        @test final_com[3] > initial_com[3]
    end

    @testset "Pure Diffusion Spreading" begin
        state_d = deepcopy(state) # Zero velocity
        
        C_in = deepcopy(state_d.tracers[:Tracer])
        C_out = state_d._buffer1[:Tracer]
        initial_peak = maximum(C_in)

        advect_diffuse_implicit_x!(C_out, C_in, state_d, grid, dt, Kh)
        final_peak = maximum(C_out)

        final_mass = sum(C_out[ng+1:nx+ng, ng+1:ny+ng, :] .* view(grid.volume, ng+1:nx+ng, ng+1:ny+ng, :))
        @test isapprox(final_mass, initial_mass, rtol=1e-7)
        @test final_peak < initial_peak
    end
end



