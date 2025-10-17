# test/test_diffusion_solver.jl

using Test
using Revise
using HydrodynamicTransport
using LinearAlgebra

# Explicitly import the internal function we are testing
using HydrodynamicTransport.VerticalTransportModule: solve_implicit_diffusion_column!

println("--- Unit Tests for Implicit Diffusion Solver (solve_implicit_diffusion_column!) on CurvilinearGrid ---")

@testset "Diffusion Solver Stability and Correctness (Curvilinear)" begin

    # --- 1. Common Setup for a Curvilinear Grid ---
    NG = 1
    nx, ny = 1, 1
    nx_tot, ny_tot = nx + 2*NG, ny + 2*NG
    
    # --- Test Case 1: Standard, well-behaved column ---
    @testset "Standard Case: Mass Conservation and Smoothing" begin
        nz = 10
        z_w = collect(range(-10.0, 0.0, length=nz+1)) # 1m thick cells
        grid = CurvilinearGrid(
            ng=NG, nx=nx, ny=ny, nz=nz,
            lon_rho=zeros(nx_tot,ny_tot), lat_rho=zeros(nx_tot,ny_tot),
            lon_u=zeros(nx_tot,ny_tot), lat_u=zeros(nx_tot,ny_tot),
            lon_v=zeros(nx_tot,ny_tot), lat_v=zeros(nx_tot,ny_tot),
            z_w=z_w, pm=ones(nx_tot,ny_tot), pn=ones(nx_tot,ny_tot),
            angle=zeros(nx_tot,ny_tot), h=ones(nx_tot,ny_tot).*10,
            mask_rho=trues(nx_tot,ny_tot), mask_u=trues(nx_tot,ny_tot), mask_v=trues(nx_tot,ny_tot),
            face_area_x=zeros(nx_tot,ny_tot,nz), face_area_y=zeros(nx_tot,ny_tot,nz),
            volume=ones(nx_tot,ny_tot,nz)
        )
        i_glob, j_glob = 1+NG, 1+NG
        dt = 60.0; Kz = 1e-4

        C_in = zeros(nz); C_in[5] = 100.0 # Sharp peak
        C_out = similar(C_in)
        
        initial_mass = sum(C_in)

        solve_implicit_diffusion_column!(C_out, C_in, grid, i_glob, j_glob, dt, Kz)
        
        final_mass = sum(C_out)

        @test isapprox(initial_mass, final_mass, rtol=1e-12) "Mass should be conserved"
        @test C_out[5] < C_in[5] "Peak should be smoothed"
        @test C_out[4] > C_in[4] && C_out[6] > C_in[6] "Adjacent cells should gain mass"
        @test !any(isnan, C_out) "Output should not contain NaNs"
    end

    # --- Test Case 2: Pathologically small dz ---
    @testset "Pathologically Small dz Case" begin
        nz = 10
        z_w = collect(range(-10.0, 0.0, length=nz+1))
        # Make one cell extremely thin
        z_w[6] = z_w[5] + 1e-10
        
        grid = CurvilinearGrid(
            ng=NG, nx=nx, ny=ny, nz=nz,
            lon_rho=zeros(nx_tot,ny_tot), lat_rho=zeros(nx_tot,ny_tot),
            lon_u=zeros(nx_tot,ny_tot), lat_u=zeros(nx_tot,ny_tot),
            lon_v=zeros(nx_tot,ny_tot), lat_v=zeros(nx_tot,ny_tot),
            z_w=z_w, pm=ones(nx_tot,ny_tot), pn=ones(nx_tot,ny_tot),
            angle=zeros(nx_tot,ny_tot), h=ones(nx_tot,ny_tot).*10,
            mask_rho=trues(nx_tot,ny_tot), mask_u=trues(nx_tot,ny_tot), mask_v=trues(nx_tot,ny_tot),
            face_area_x=zeros(nx_tot,ny_tot,nz), face_area_y=zeros(nx_tot,ny_tot,nz),
            volume=ones(nx_tot,ny_tot,nz)
        )
        i_glob, j_glob = 1+NG, 1+NG
        dt = 60.0; Kz = 1e-4

        C_in = zeros(nz); C_in[5] = 100.0
        C_out = similar(C_in)

        solve_implicit_diffusion_column!(C_out, C_in, grid, i_glob, j_glob, dt, Kz)
        
        @test !any(isnan, C_out) "Solver must not produce NaNs even with tiny dz"
    end
    
    # --- Test Case 3: Single Cell ---
    @testset "Single Cell Case" begin
        nz = 1
        z_w = [-1.0, 0.0]
        grid = CurvilinearGrid(
            ng=NG, nx=nx, ny=ny, nz=nz,
            lon_rho=zeros(nx_tot,ny_tot), lat_rho=zeros(nx_tot,ny_tot),
            lon_u=zeros(nx_tot,ny_tot), lat_u=zeros(nx_tot,ny_tot),
            lon_v=zeros(nx_tot,ny_tot), lat_v=zeros(nx_tot,ny_tot),
            z_w=z_w, pm=ones(nx_tot,ny_tot), pn=ones(nx_tot,ny_tot),
            angle=zeros(nx_tot,ny_tot), h=ones(nx_tot,ny_tot).*10,
            mask_rho=trues(nx_tot,ny_tot), mask_u=trues(nx_tot,ny_tot), mask_v=trues(nx_tot,ny_tot),
            face_area_x=zeros(nx_tot,ny_tot,nz), face_area_y=zeros(nx_tot,ny_tot,nz),
            volume=ones(nx_tot,ny_tot,nz)
        )
        i_glob, j_glob = 1+NG, 1+NG
        dt = 60.0; Kz = 1e-4

        C_in = [100.0]
        C_out = similar(C_in)

        solve_implicit_diffusion_column!(C_out, C_in, grid, i_glob, j_glob, dt, Kz)
        
        @test C_in == C_out "Single cell column should not change"
    end

end

println("\n✅ All Curvilinear diffusion solver unit tests complete.")