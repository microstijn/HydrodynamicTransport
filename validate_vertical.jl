# validate_vertical.jl
#
# Group B — vertical-transport validation: the diagnosed-omega + implicit upwind/CN path.
#   B1  constancy preservation under diagnosed omega (vs w ≡ 0)
#   B2  1-D vertical advection order study (implicit upwind -> p ≈ 1)
#   B3  high-Courant stability of the implicit vertical solve
# Prints tables and writes vertical_validation_results.csv. Self-contained.
#
#   julia --project=. validate_vertical.jl

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "test", "benchmarks", "benchmark_common.jl"))
using .BenchmarkCommon
include(joinpath(@__DIR__, "test", "benchmarks", "vertical_benchmarks.jl"))

sig(x; d=4) = x isa Real ? string(round(Float64(x); sigdigits=d)) : string(x)
pad(x, n) = rpad(sig(x), n)
csv_rows = Any[]

println("="^78)
println("GROUP B — VERTICAL TRANSPORT (diagnosed omega + implicit solve)")
println("="^78)

# ------------------------------------------------------- B1 continuity closure
println("\nB1  Continuity closure of the diagnosed omega: max relative net-flux divergence over")
println("    interior cells for a depth-integrated non-divergent flow (should be ~machine 0)\n")
c = bench_continuity_closure(; nx=20, nz=8)
println("    diagnosed omega : ", sig(c.residual_diagnosed; d=3), "   (continuity closes)")
println("    w ≡ 0           : ", sig(c.residual_zero_w; d=3), "   (uncompensated HDiv)")
push!(csv_rows, ["B1_closure_diagnosed", c.residual_diagnosed])
push!(csv_rows, ["B1_closure_zero_w", c.residual_zero_w])

# ------------------------------------------------------- B2 vertical advection order
println("\nB2  1-D vertical advection of a Gaussian (constant interior w) — order of accuracy\n")
println(pad("nz", 6), pad("dz", 11), pad("L2", 14), pad("min", 12), pad("max", 12))
rows = bench_vertical_advection(; resolutions=[40, 80, 160, 320], W=1.0, courant=0.4)
for r in rows
    println(pad(r.nz, 6), pad(r.dz, 11), pad(r.L2, 14), pad(r.minval, 12), pad(r.maxval, 12))
    push!(csv_rows, ["B2_vadvection", r.nz, r.dz, r.dt, r.L2, r.minval, r.maxval])
end
println("   -> empirical convergence order p(L2) = ",
        sig(fit_order([r.dz for r in rows], [r.L2 for r in rows])), "  (expect ≈ 1, implicit upwind)\n")

# ------------------------------------------------------- B3 high-Courant stability
println("\nB3  High-Courant stability of the implicit vertical advection\n")
s = bench_vertical_stability(; nz=40, courant=5.0, nsteps=200, W=1.0)
println("    vertical Courant : ", sig(s.max_courant))
println("    finite           : ", s.finite)
println("    min / max        : ", sig(s.minval), " / ", sig(s.maxval))
push!(csv_rows, ["B3_stability_courant", s.max_courant])
push!(csv_rows, ["B3_stability_min", s.minval])
push!(csv_rows, ["B3_stability_max", s.maxval])

out = joinpath(@__DIR__, "vertical_validation_results.csv")
BenchmarkCommon.write_results_csv(out, ["case", "v1", "v2", "v3", "v4", "v5", "v6"], csv_rows)
println("\nWrote ", out)
