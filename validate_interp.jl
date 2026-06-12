# Regression for optimization #1: update_hydrodynamics! must return identical interpolated
# values with the slab cache, including clamping and NON-monotonic time access (which
# exercises cache hits, slab eviction, and re-loading a previously-seen index).
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.HydrodynamicsModule: update_hydrodynamics!
using NCDatasets
using Printf

ok = true
mktempdir() do dir
    filename = joinpath(dir, "test_hydro.nc")
    ds = NCDataset(filename, "c")
    defDim(ds, "xi_u", 1); defDim(ds, "eta_u", 1); defDim(ds, "s_rho", 1); defDim(ds, "ocean_time", 2)
    defVar(ds, "ocean_time", [0.0, 10.0], ("ocean_time",))
    uv = defVar(ds, "u", Float64, ("xi_u", "eta_u", "s_rho", "ocean_time"))
    uv[:,:,:,1] = fill(1.0, (1,1,1)); uv[:,:,:,2] = fill(3.0, (1,1,1))
    close(ds)

    ng = 2
    grid = CurvilinearGrid(ng, 1, 1, 1, zeros(1+2ng,1+2ng), zeros(1+2ng,1+2ng), zeros(1-1+2ng,1+2ng), zeros(1-1+2ng,1+2ng),
        zeros(1+2ng,1-1+2ng), zeros(1+2ng,1-1+2ng), [-1.0,0.0], ones(1+2ng,1+2ng), ones(1+2ng,1+2ng), zeros(1+2ng,1+2ng),
        ones(1+2ng,1+2ng), trues(1+2ng,1+2ng), trues(1-1+2ng,1+2ng), trues(1+2ng,1-1+2ng),
        zeros(1+1+2ng,1+2ng,1), zeros(1+2ng,1+1+2ng,1), zeros(1+2ng,1+2ng,1))
    state = initialize_state(grid, ())
    ds = NCDataset(filename)
    hydro = HydrodynamicData(filename, Dict(:u => "u", :time => "ocean_time"))

    check(t, expect) = begin
        update_hydrodynamics!(state, grid, ds, hydro, t)
        up = state.u[ng+1:grid.nx+ng, ng+1:grid.ny+ng, :]
        pass = all(isapprox.(up, expect))
        @printf("  t=%6.1f -> u=%.3f (expect %.1f)  %s\n", t, up[1], expect, pass ? "OK" : "FAIL")
        global ok &= pass
    end
    println("Interpolation/clamping with slab cache (non-monotonic time):")
    check(5.0, 2.0)    # midpoint: loads idx 1 & 2
    check(0.0, 1.0)    # exact start: idx 1 (cache hit)
    check(-1.0, 1.0)   # clamp before start
    check(11.0, 3.0)   # clamp after end: idx 2 (cache hit, backward-then-forward access)
    check(5.0, 2.0)    # back to midpoint again
    close(ds)
end
println(ok ? "\n✅ interpolation cache regression PASSED" : "\n❌ interpolation cache regression FAILED")
exit(ok ? 0 : 1)
