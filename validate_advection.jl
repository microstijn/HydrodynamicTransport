# validate_advection.jl
#
# Group A — analytical horizontal-advection validation of the tracer solver. Runs the standard
# benchmark flows (uniform translation order study, solid-body Gaussian rotation, Zalesak slotted
# cylinder) for the three schemes (:FFSL production, :TVD, :UP3), prints metric tables and writes
# advection_validation_results.csv. Self-contained (synthetic grids); a few minutes single-threaded.
#
#   julia --project=. validate_advection.jl
#   julia -t auto --project=. validate_advection.jl     # tracer-parallel (1 tracer here, so serial)

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "test", "benchmarks", "benchmark_common.jl"))
using .BenchmarkCommon
include(joinpath(@__DIR__, "test", "benchmarks", "advection_benchmarks.jl"))

sig(x; d=4) = x isa Real ? string(round(Float64(x); sigdigits=d)) : string(x)
pad(x, n) = rpad(sig(x), n)
const SCHEMES = [:FFSL, :TVD, :UP3]
const RES = [40, 80, 160]

csv_rows = Any[]

println("="^78)
println("GROUP A — HORIZONTAL ADVECTION (analytical flows)")
println("="^78)

# ---------------------------------------------------------------- A1: order of accuracy
println("\nA1  Uniform translation of a Gaussian — order of accuracy (L2 vs dx)\n")
println(pad("scheme", 8), pad("nx", 6), pad("dx", 10), pad("L2", 14), pad("mass_drift", 14))
for scheme in SCHEMES
    rows = bench_translation(scheme; resolutions=RES, U=1.0, courant=0.5)
    for r in rows
        println(pad(scheme, 8), pad(r.nx, 6), pad(r.dx, 10), pad(r.L2, 14), pad(r.mass_drift, 14))
        push!(csv_rows, ["A1_translation", scheme, r.nx, r.dx, r.dt, r.L1, r.L2, r.Linf,
                         r.mass_drift, r.minval, r.maxval, r.peak_retention])
    end
    p = fit_order([r.dx for r in rows], [r.L2 for r in rows])
    println("   -> empirical convergence order p(L2) = ", sig(p), "\n")
end

# ---------------------------------------------------------------- A2: rotation
println("\nA2  Solid-body rotation of a Gaussian hill — 1 revolution (exact = IC)\n")
println(pad("scheme", 8), pad("Courant", 9), pad("L2", 13), pad("Linf", 13),
        pad("peak_ret", 11), pad("min", 12), pad("mass_drift", 13))
for scheme in SCHEMES
    r = bench_gaussian_rotation(scheme; nx=120, courant=0.5)
    println(pad(scheme, 8), pad(0.5, 9), pad(r.L2, 13), pad(r.Linf, 13),
            pad(r.peak_retention, 11), pad(r.minval, 12), pad(r.mass_drift, 13))
    push!(csv_rows, ["A2_rotation", scheme, r.nx, r.dx, r.dt, r.L1, r.L2, r.Linf,
                     r.mass_drift, r.minval, r.maxval, r.peak_retention])
end
# FFSL large-Courant capability
rhi = bench_gaussian_rotation(:FFSL; nx=120, courant=3.0)
println(pad(:FFSL, 8), pad(3.0, 9), pad(rhi.L2, 13), pad(rhi.Linf, 13),
        pad(rhi.peak_retention, 11), pad(rhi.minval, 12), pad(rhi.mass_drift, 13))
push!(csv_rows, ["A2_rotation_Co3", :FFSL, rhi.nx, rhi.dx, rhi.dt, rhi.L1, rhi.L2, rhi.Linf,
                 rhi.mass_drift, rhi.minval, rhi.maxval, rhi.peak_retention])

# ---------------------------------------------------------------- A3: Zalesak
println("\nA3  Zalesak slotted cylinder — 1 revolution (monotonicity: min/max under/overshoot)\n")
println(pad("scheme", 8), pad("L1", 13), pad("min", 13), pad("max", 13), pad("mass_drift", 13))
for scheme in SCHEMES
    r = bench_zalesak(scheme; nx=120, courant=0.5)
    println(pad(scheme, 8), pad(r.L1, 13), pad(r.minval, 13), pad(r.maxval, 13), pad(r.mass_drift, 13))
    push!(csv_rows, ["A3_zalesak", scheme, r.nx, r.dx, r.dt, r.L1, r.L2, r.Linf,
                     r.mass_drift, r.minval, r.maxval, r.peak_retention])
end

header = ["case", "scheme", "nx", "dx", "dt", "L1", "L2", "Linf",
          "mass_drift", "min", "max", "peak_retention"]
out = joinpath(@__DIR__, "advection_validation_results.csv")
BenchmarkCommon.write_results_csv(out, header, csv_rows)
println("\nWrote ", out)
