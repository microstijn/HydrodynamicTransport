# --- 1. Set up the Environment ---
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using NCDatasets
using Test

# --- 2. Configuration ---
const NG = 2
const NC_URL = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"

println("--- Test Script for initialize_curvilinear_grid ---")
println("This script verifies that grid data is correctly loaded into a ghost-cell-aware structure.")
println("Using ng = $NG ghost cells.")

# --- 3. Run the Tests ---
# Wrap in a try-catch block to handle potential network errors
try
    @testset "Curvilinear Grid Initialization with Ghost Cells" begin
        
        grid = initialize_curvilinear_grid(NC_URL, ng = 2)
        ds = NCDataset(NC_URL)
        
        # --- Test 1: Check Struct Properties ---
        @testset "Struct Properties" begin
            @test grid.ng == NG
            @test grid.nx == ds.dim["xi_rho"]
            @test grid.ny == ds.dim["eta_rho"]
            println("✅ Grid struct properties (ng, nx, ny) are correct.")
        end
        
        # --- Test 2: Check Array Sizing ---
        @testset "Array Sizing" begin
            nx_phys, ny_phys = ds.dim["xi_rho"], ds.dim["eta_rho"]
            nx_tot, ny_tot = nx_phys + 2*NG, ny_phys + 2*NG
            
            @test size(grid.pm) == (nx_tot, ny_tot)
            @test size(grid.lon_rho) == (nx_tot, ny_tot)
            println("✅ Rho-point metric arrays have the correct total size.")

            nx_u_phys, ny_u_phys = ds.dim["xi_u"], ds.dim["eta_u"]
            @test size(grid.lon_u) == (nx_u_phys + 2*NG, ny_u_phys + 2*NG)
            println("✅ U-point metric arrays have the correct total size.")
        end

        # --- Test 3: Data Loading into Interior ---
        @testset "Data Loading into Interior" begin
            nx_phys, ny_phys = ds.dim["xi_rho"], ds.dim["eta_rho"]
            
            pm_from_file = ds["pm"][:,:]
            pm_interior = view(grid.pm, NG+1:nx_phys+NG, NG+1:ny_phys+NG)
            @test pm_interior == pm_from_file
            println("✅ Physical data was correctly loaded into the array interior.")
        end

        # --- Test 4: Extrapolation into Ghost Cells ---
        @testset "Metric Extrapolation" begin
            nx_phys, ny_phys = ds.dim["xi_rho"], ds.dim["eta_rho"]
            nx_tot, ny_tot = nx_phys + 2*NG, ny_phys + 2*NG

            # Check a corner value
            @test grid.pm[1, 1] != 0.0
            @test grid.pm[1, 1] == grid.pm[NG+1, NG+1] # SW corner
            println("✅ South-West corner was correctly extrapolated.")

            # Check another corner
            @test grid.pm[nx_tot, ny_tot] != 0.0
            @test grid.pm[nx_tot, ny_tot] == grid.pm[nx_phys+NG, ny_phys+NG] # NE corner
            println("✅ North-East corner was correctly extrapolated.")

            # Check a side value
            mid_j = div(ny_tot, 2)
            @test grid.pm[1, mid_j] != 0.0
            @test grid.pm[1, mid_j] == grid.pm[NG+1, mid_j] # West side
            println("✅ West side was correctly extrapolated.")
        end
        
        close(ds)
    end
catch e
    if isa(e, NCDatasets.NetCDFError)
        @warn "Skipping test: Could not access remote NetCDF data at '$NC_URL'. Server may be temporarily unavailable."
    else
        rethrow(e)
    end
end

println("\n--- Test Script Finished ---")