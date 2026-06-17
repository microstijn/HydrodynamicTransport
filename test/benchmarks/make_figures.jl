# test/benchmarks/make_figures.jl
#
# Render the solver-validation figures for the paper with CairoMakie. Kept out of the package deps
# (the package itself needs no plotting); run in an environment that has BOTH CairoMakie and this
# package available, e.g.:
#
#   julia --project=<env-with-CairoMakie-and-HydrodynamicTransport> test/benchmarks/make_figures.jl [outdir]
#
# It re-runs the headline studies (a few minutes) rather than parsing the result CSVs, and writes:
#   solver_validation_convergence.pdf  — L2-vs-spacing order studies (advection / vertical / diffusion)
#   solver_validation_schemes.pdf      — rotation/Zalesak scheme comparison (peak retention, positivity)
# Copy the PDFs into previr-paper/figures/.

using CairoMakie
CairoMakie.activate!(type="pdf")

const HERE = @__DIR__
include(joinpath(HERE, "benchmark_common.jl"));   using .BenchmarkCommon
include(joinpath(HERE, "advection_benchmarks.jl"))
include(joinpath(HERE, "vertical_benchmarks.jl"))
include(joinpath(HERE, "diffusion_benchmarks.jl"))

# Default output: the sibling paper repo's figures/ dir (…/programming/previr-paper/figures);
# override with a path argument.
outdir = length(ARGS) >= 1 ? ARGS[1] :
         normpath(joinpath(HERE, "..", "..", "..", "previr-paper", "figures"))
mkpath(outdir)

# reference power-law guide line through the first point, slope p
guide(h, e0, h0, p) = e0 .* (h ./ h0) .^ p

println("Running convergence studies for figures …")
trF = bench_translation(:FFSL; resolutions=[40, 80, 160], U=1.0, courant=0.5)
vb  = bench_vertical_advection(; resolutions=[40, 80, 160, 320], W=1.0, courant=0.4)
hd  = bench_horizontal_diffusion(; resolutions=[40, 80, 160], Kh=2.0)
vd  = bench_vertical_diffusion(; resolutions=[20, 40, 80, 160], Kz=1e-3)

# ---------------- Figure 1: convergence (order of accuracy) ----------------
fig1 = Figure(size=(1080, 360))

ax1 = Axis(fig1[1, 1]; xscale=log10, yscale=log10, xlabel="grid spacing Δx", ylabel="L2 error",
           title="horizontal advection")
let h = [r.dx for r in trF], e = [r.L2 for r in trF]
    scatterlines!(ax1, h, e; label="advection")
    lines!(ax1, h, guide(h, e[1], h[1], 2); linestyle=:dash, color=:gray, label="slope 2")
end
axislegend(ax1; position=:rb)

ax2 = Axis(fig1[1, 2]; xscale=log10, yscale=log10, xlabel="layer thickness Δz", ylabel="L2 error",
           title="vertical advection")
let h = [r.dz for r in vb], e = [r.L2 for r in vb]
    scatterlines!(ax2, h, e; label="implicit upwind")
    lines!(ax2, h, guide(h, e[1], h[1], 1); linestyle=:dash, color=:gray, label="slope 1")
end
axislegend(ax2; position=:rb)

ax3 = Axis(fig1[1, 3]; xscale=log10, yscale=log10, xlabel="grid spacing", ylabel="L2 error",
           title="diffusion")
let h = [r.dx for r in hd], e = [r.L2 for r in hd]
    scatterlines!(ax3, h, e; label="horizontal")
    lines!(ax3, h, guide(h, e[1], h[1], 2); linestyle=:dash, color=:gray, label="slope 2")
end
scatterlines!(ax3, [r.dz for r in vd], [r.L2 for r in vd]; label="vertical (CN)")
axislegend(ax3; position=:rb)

save(joinpath(outdir, "solver_validation_convergence.pdf"), fig1)
println("wrote ", joinpath(outdir, "solver_validation_convergence.pdf"))
