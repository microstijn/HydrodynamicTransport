# test/test_adsorption_desorption.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test

println("--- Unit Test for Adsorption/Desorption Function ---")

# --- 1. Define the function to be tested locally ---
# We copy the exact function from the simulation script for a direct test.
function adsorption_desorption(concentrations, environment, dt)
    C_diss = max(0.0, concentrations[:Virus_Dissolved])
    C_sorb = max(0.0, concentrations[:Virus_Sorbed])
    TSS = environment.TSS

    Kd = 0.2
    transfer_rate = 0.0001

    C_sorb_eq = Kd * TSS * C_diss
    delta_C = (C_sorb_eq - C_sorb) * transfer_rate * dt

    if delta_C > 0 # Adsorption (dissolved -> sorbed)
        delta_C = min(delta_C, C_diss)
    else # Desorption (sorbed -> dissolved)
        delta_C = max(delta_C, -C_sorb)
    end
    
    return Dict(:Virus_Dissolved => -delta_C, :Virus_Sorbed => +delta_C)
end

# --- 2. Run the Test Suite ---
@testset "Adsorption/Desorption Kinetics" begin

    # --- Shared test parameters ---
    mock_env = (TSS = 5.0,) # Mock environment with 5.0 g/m^3 of TSS
    dt = 60.0 # 60 second time step

    @testset "Case 1: Pure Adsorption" begin
        # Start with only dissolved virus
        C_initial = Dict(:Virus_Dissolved => 100.0, :Virus_Sorbed => 0.0)
        
        dC = adsorption_desorption(C_initial, mock_env, dt)

        @test dC[:Virus_Dissolved] < 0
        @test dC[:Virus_Sorbed] > 0
        @test isapprox(dC[:Virus_Dissolved], -dC[:Virus_Sorbed]) # Perfect mass balance
        @test !any(isnan, values(dC))
    end

    @testset "Case 2: Pure Desorption" begin
        # Start with only sorbed virus
        C_initial = Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 100.0)
        
        dC = adsorption_desorption(C_initial, mock_env, dt)

        @test dC[:Virus_Dissolved] > 0
        @test dC[:Virus_Sorbed] < 0
        @test isapprox(dC[:Virus_Dissolved], -dC[:Virus_Sorbed]) # Perfect mass balance
        @test !any(isnan, values(dC))
    end

    @testset "Case 3: At Equilibrium" begin
        # Start at equilibrium: C_sorb = Kd * TSS * C_diss = 0.2 * 5.0 * 10.0 = 10.0
        C_initial = Dict(:Virus_Dissolved => 10.0, :Virus_Sorbed => 10.0)
        
        dC = adsorption_desorption(C_initial, mock_env, dt)

        @test isapprox(dC[:Virus_Dissolved], 0.0, atol=1e-12)
        @test isapprox(dC[:Virus_Sorbed], 0.0, atol=1e-12)
        @test !any(isnan, values(dC))
    end

    @testset "Case 4: Clamping Logic (Adsorption)" begin
        # Create a case where the calculated change would exceed the available mass
        # High transfer_rate and dt will try to transfer more than 100.0 units
        C_initial = Dict(:Virus_Dissolved => 100.0, :Virus_Sorbed => 0.0)
        fast_dt = 100000.0 # A very large dt
        
        dC = adsorption_desorption(C_initial, mock_env, fast_dt)

        # The change should be clamped to not remove more than the initial 100.0
        @test dC[:Virus_Dissolved] == -100.0
        @test dC[:Virus_Sorbed] == 100.0
        @test !any(isnan, values(dC))
    end

    @testset "Case 5: Clamping Logic (Desorption)" begin
        # Create a case where the calculated change would make sorbed concentration negative
        C_initial = Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 50.0)
        fast_dt = 100000.0 # A very large dt

        dC = adsorption_desorption(C_initial, mock_env, fast_dt)

        # The change should be clamped to not remove more than the initial 50.0
        @test dC[:Virus_Dissolved] == 50.0
        @test dC[:Virus_Sorbed] == -50.0
        @test !any(isnan, values(dC))
    end
    
    @testset "Case 6: Zero Initial Concentrations" begin
        C_initial = Dict(:Virus_Dissolved => 0.0, :Virus_Sorbed => 0.0)
        
        dC = adsorption_desorption(C_initial, mock_env, dt)

        @test dC[:Virus_Dissolved] == 0.0
        @test dC[:Virus_Sorbed] == 0.0
        @test !any(isnan, values(dC))
    end

end

println("\n✅ All adsorption/desorption unit tests passed.")