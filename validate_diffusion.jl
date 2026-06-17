# validate_diffusion.jl
#
# Group C — diffusion validation against the analytical Gaussian-spreading (heat-equation) solution.
#   C1  2-D horizontal diffusion order study (explicit central -> p ≈ 2) + variance + mass
#   C2  1-D vertical CN diffusion order study (-> p ≈ 2) + variance
#   C3  vertical CN stability at large diffusion number
# Prints tables and writes diffusion_validation_results.csv. Self-contained.
#
#   julia --project=. validate_diffusion.jl

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "test", "benchmarks", "benchmark_common.jl"))
using .BenchmarkCommon
include(joinpath(@__DIR__, "test", "benchmarks", "diffusion_benchmarks.jl"))

sig(x; d=4) = x isa Real ? string(round(Float64(x); sigdigits=d)) : string(x)
pad(x, n) = rpad(sig(x), n)
csv_rows = Any[]

println("="^78)
println("GROUP C — DIFFUSION (analytical Gaussian spreading)")
println("="^78)

# ------------------------------------------------------- C1 horizontal diffusion
println("\nC1  2-D horizontal diffusion of a Gaussian — order of accuracy (L2 vs dx)\n")
println(pad("nx", 6), pad("dx", 10), pad("L2", 13), pad("σ²_num", 12), pad("σ²_exact", 12),
        pad("σ²_relerr", 12), pad("mass_drift", 13))
crows = bench_horizontal_diffusion(; resolutions=[40, 80, 160], Kh=2.0)
for r in crows
    println(pad(r.nx, 6), pad(r.dx, 10), pad(r.L2, 13), pad(r.sigma2_num, 12),
            pad(r.sigma2_exact, 12), pad(r.sigma2_relerr, 12), pad(r.mass_drift, 13))
    push!(csv_rows, ["C1_hdiffusion", r.nx, r.dx, r.dt, r.L2, r.sigma2_num, r.sigma2_relerr, r.mass_drift])
end
println("   -> empirical convergence order p(L2) = ",
        sig(fit_order([r.dx for r in crows], [r.L2 for r in crows])), "  (expect ≈ 2)\n")

# ------------------------------------------------------- C2 vertical CN diffusion
println("\nC2  1-D vertical CN diffusion of a Gaussian — order of accuracy (L2 vs dz)\n")
println(pad("nz", 6), pad("dz", 11), pad("L2", 13), pad("σ²_num", 12), pad("σ²_relerr", 12))
vrows = bench_vertical_diffusion(; resolutions=[20, 40, 80, 160], Kz=1e-3)
for r in vrows
    println(pad(r.nz, 6), pad(r.dz, 11), pad(r.L2, 13), pad(r.sigma2_num, 12), pad(r.sigma2_relerr, 12))
    push!(csv_rows, ["C2_vdiffusion", r.nz, r.dz, r.dt, r.L2, r.sigma2_num, r.sigma2_relerr, 0.0])
end
println("   -> empirical convergence order p(L2) = ",
        sig(fit_order([r.dz for r in vrows], [r.L2 for r in vrows])), "  (expect ≈ 2)\n")

# ------------------------------------------------------- C3 vertical CN stability
println("\nC3  Vertical CN stability at large diffusion number (Kz·dt/dz² ≫ 1)\n")
s = bench_vertical_diffusion_stability(; nz=40, diffnum=10.0, nsteps=50, Kz=1e-3)
println("    diffusion number : ", sig(s.diffnum))
println("    finite           : ", s.finite)
println("    min / max        : ", sig(s.minval), " / ", sig(s.maxval))
push!(csv_rows, ["C3_stability_diffnum", s.diffnum, 0, 0, 0, s.minval, s.maxval, 0])

out = joinpath(@__DIR__, "diffusion_validation_results.csv")
BenchmarkCommon.write_results_csv(out,
    ["case", "n", "h", "dt", "L2", "sigma2_num", "sigma2_relerr", "mass_drift"], csv_rows)
println("\nWrote ", out)
