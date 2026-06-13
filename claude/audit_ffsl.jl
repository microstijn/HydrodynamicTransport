# Type-stability audit of the FFSL kernels on the REAL CurviLoire grid.
# Handoff item #1: untyped args (C_out, C_in, dt, ...) might box / force dynamic dispatch.
# This captures code_warntype text for each kernel and flags non-concrete inferred types.
# Run:  JULIA_NUM_THREADS=8 julia +nightly claude/audit_ffsl.jl
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using InteractiveUtils
using NCDatasets

const HTM = HydrodynamicTransport.HorizontalTransportModule

const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"

println("Loading real CurviLoire grid + state ...")
grid = initialize_curvilinear_grid(NC)
ds = NCDataset(NC)
state = initialize_state(grid, ds, (:A, :B))
close(ds)

dt = 60.0; D_crit = 0.05
C_in  = state.tracers[:A]
C_out = state._buffer1[:A]
nx, ny, nz = grid.nx, grid.ny, grid.nz
ng = grid.ng
m  = nx + 2*ng

# scratch for the line kernel
mk(n) = Vector{Float64}(undef, n)
crow, vrow, cnew = mk(m), mk(m), mk(m)
cL, cR, Flo, Fhi, Ctd, Rp, Rm = mk(m), mk(m), mk(m), mk(m), mk(m), mk(m), mk(m)
crf = view(state.flux_x, 1:m, 1+ng, 1)   # Float32 SubArray, exactly as the real call site

# Capture code_warntype text and scan for red flags.
function audit(name, f, types)
    io = IOBuffer()
    code_warntype(io, f, types)
    txt = String(take!(io))
    # markers of instability in code_warntype text output
    bad = String[]
    for pat in ("::Any", "::Union{", "Box(", "Core.Box")
        occursin(pat, txt) && push!(bad, pat)
    end
    status = isempty(bad) ? "OK (type-stable)" : "FLAGGED: " * join(unique(bad), ", ")
    println("\n", "="^78)
    println("### ", name, "  ->  ", status)
    println("="^78)
    # Always show the Body return type line + any flagged lines for the record
    for ln in split(txt, '\n')
        if occursin("Body::", ln) || occursin("::Any", ln) || occursin("::Union{", ln) || occursin("Box", ln)
            println(ln)
        end
    end
    return isempty(bad)
end

results = Dict{String,Bool}()
results["advect_x_ffsl!"] = audit("advect_x_ffsl!", HTM.advect_x_ffsl!,
    Tuple{typeof(C_out), typeof(C_in), typeof(state), typeof(grid), typeof(dt), Float64})
results["advect_y_ffsl!"] = audit("advect_y_ffsl!", HTM.advect_y_ffsl!,
    Tuple{typeof(C_out), typeof(C_in), typeof(state), typeof(grid), typeof(dt), Float64})
results["_ffsl_line!"] = audit("_ffsl_line!", HTM._ffsl_line!,
    Tuple{typeof(cnew),typeof(crow),typeof(vrow),typeof(crf),typeof(cL),typeof(cR),
          typeof(Flo),typeof(Fhi),typeof(Ctd),typeof(Rp),typeof(Rm),Int,Int,Int})
results["_ffsl_ppm_edges!"] = audit("_ffsl_ppm_edges!", HTM._ffsl_ppm_edges!,
    Tuple{typeof(cL),typeof(cR),typeof(crow),Int})
results["_ffsl_face_flux"] = audit("_ffsl_face_flux", HTM._ffsl_face_flux,
    Tuple{typeof(cL),typeof(cR),typeof(crow),typeof(vrow),Int,Int,Float64})
results["_ffsl_face_flux_low"] = audit("_ffsl_face_flux_low", HTM._ffsl_face_flux_low,
    Tuple{typeof(crow),typeof(vrow),Int,Int,Float64})
results["_compute_face_courant!"] = audit("_compute_face_courant!", HTM._compute_face_courant!,
    Tuple{typeof(state), typeof(grid), Float64, Float64})

println("\n", "#"^78)
println("SUMMARY")
for (k,v) in results
    println(rpad(k, 28), v ? "OK" : "FLAGGED")
end
println("#"^78)
