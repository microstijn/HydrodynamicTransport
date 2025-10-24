using Pkg

using Test
include("../src/FluxLimiters.jl")

@testset "Flux Limiters Tests" begin
    @testset "van_leer" begin
        # Test cases for the van_leer flux limiter function

        # When r is positive and less than 1, it should return a value between 0 and 1
        @test isapprox(FluxLimiters.van_leer(0.5), 2 * 0.5 / (1 + 0.5))

        # When r is positive and greater than 1, it should also return a value between 1 and 2
        @test isapprox(FluxLimiters.van_leer(1.5), 2 * 1.5 / (1 + 1.5))

        # When r is negative, it should return 0
        @test FluxLimiters.van_leer(-0.5) == 0.0
        @test FluxLimiters.van_leer(-1.0) == 0.0

        # When r is zero, it should return 0
        @test FluxLimiters.van_leer(0.0) == 0.0

        # When r is exactly 1, it should return 1
        @test FluxLimiters.van_leer(1.0) == 1.0
    end

    @testset "calculate_limited_flux" begin
        velocity = 1.0
        face_area = 1.0

        # Case 1: Smooth, linear gradient (r=1) -> should be central-difference
        c_up_far, c_up_near, c_down_near = 0.0, 1.0, 2.0
        expected_flux = velocity * 1.5 * face_area # Central difference result
        @test isapprox(FluxLimiters.calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area), expected_flux)

        # Case 2: Oscillatory region (a peak, r=-1) -> should revert to upwind
        c_up_far, c_up_near, c_down_near = 0.0, 1.0, 0.0
        expected_flux = velocity * c_up_near * face_area # Upwind result
        @test isapprox(FluxLimiters.calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area), expected_flux)

        # Case 3: Approaching a plateau/extremum (denominator is zero) -> should be handled gracefully
        c_up_far, c_up_near, c_down_near = 0.0, 1.0, 1.0
        expected_flux = velocity * c_up_near * face_area # Upwind result
        @test isapprox(FluxLimiters.calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area), expected_flux)

        # Case 4: A less smooth region where 0 < r < 1
        c_up_far, c_up_near, c_down_near = 0.5, 1.0, 2.0
        # r = (1.0 - 0.5) / (2.0 - 1.0) = 0.5
        # phi = (0.5 + 0.5) / (1 + 0.5) = 1 / 1.5 = 2/3
        # c_face = 1.0 + 0.5 * (2/3) * (2.0 - 1.0) = 1.0 + 1/3 = 4/3
        expected_flux = velocity * (4/3) * face_area
        @test isapprox(FluxLimiters.calculate_limited_flux(c_up_far, c_up_near, c_down_near, velocity, face_area), expected_flux)

        # Case 5: Negative velocity (flow from right to left)
        # The logic inside calculate_limited_flux is independent of velocity sign,
        # it just multiplies by it at the end. The caller is responsible for providing the correct stencil.
        c_up_far, c_up_near, c_down_near = 2.0, 1.0, 0.0 # Stencil for flow R->L
        neg_velocity = -1.0
        # r = (1.0 - 2.0) / (0.0 - 1.0) = -1.0 / -1.0 = 1.0
        # phi = 1.0
        # c_face = 1.0 + 0.5 * 1.0 * (0.0 - 1.0) = 0.5
        expected_flux = neg_velocity * 0.5 * face_area
        @test isapprox(FluxLimiters.calculate_limited_flux(c_up_far, c_up_near, c_down_near, neg_velocity, face_area), expected_flux)
    end
end