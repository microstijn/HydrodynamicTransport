# Same-process A/B for the diffusion restructure (option 2): OLD full-flux-buffer version
# (held locally) vs NEW per-line-scratch version (HTM.diffuse_{x,y}! from source). Interleaved
# timing cancels thermal drift; also verifies the two produce bit-identical output.
#   JULIA_NUM_THREADS=8 julia +nightly claude/bench_diff_ab.jl
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")
using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs, HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule, HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.HydrodynamicsModule
using NCDatasets, Printf
const HT = HydrodynamicTransport
const HTM = HydrodynamicTransport.HorizontalTransportModule
const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"
const DT=60.0; const DC=0.05; const KH=1.0

# ---- OLD: full 3-D flux buffer (verbatim original) ----
function old_diffuse_x!(C_out, C_in, state, grid, dt, Kh, fx, D_crit)
    nx, ny = grid.nx, grid.ny; ng = grid.ng
    fx .= 0.0
    @inbounds for k in axes(C_in,3)
        for j_phys in 1:ny, i_phys in 2:nx
            ig, jg = i_phys+ng, j_phys+ng; flux = 0.0
            if isa(grid, CurvilinearGrid)
                d1 = grid.h[ig-1,jg]+state.zeta[ig-1,jg,k]; d2 = grid.h[ig,jg]+state.zeta[ig,jg,k]
                if d1 < D_crit || d2 < D_crit; flux = 0.0
                else; dx = HTM.get_dx_centers(grid,ig,jg); flux = -Kh*grid.face_area_x[ig,jg,k]*((C_in[ig,jg,k]-C_in[ig-1,jg,k])/dx); end
            else; dx = HTM.get_dx_centers(grid,ig,jg); flux = -Kh*grid.face_area_x[ig,jg,k]*((C_in[ig,jg,k]-C_in[ig-1,jg,k])/dx); end
            w = isa(grid,CurvilinearGrid) ? grid.mask_u[ig,jg] : (grid.mask[ig,jg,k] & grid.mask[ig-1,jg,k])
            fx[ig,jg,k] = flux*w
        end
    end
    @inbounds for k in axes(C_out,3), j_phys in 1:ny, i_phys in 1:nx
        ig, jg = i_phys+ng, j_phys+ng
        C_out[ig,jg,k] = C_in[ig,jg,k] - (dt/grid.volume[ig,jg,k])*(fx[ig+1,jg,k]-fx[ig,jg,k])
    end
end
function old_diffuse_y!(C_out, C_in, state, grid, dt, Kh, fy, D_crit)
    nx, ny = grid.nx, grid.ny; ng = grid.ng
    fy .= 0.0
    @inbounds for k in axes(C_in,3)
        for j_phys in 2:ny, i_phys in 1:nx
            ig, jg = i_phys+ng, j_phys+ng; flux = 0.0
            if isa(grid, CurvilinearGrid)
                d1 = grid.h[ig,jg-1]+state.zeta[ig,jg-1,k]; d2 = grid.h[ig,jg]+state.zeta[ig,jg,k]
                if d1 < D_crit || d2 < D_crit; flux = 0.0
                else; dy = HTM.get_dy_centers(grid,ig,jg); flux = -Kh*grid.face_area_y[ig,jg,k]*((C_in[ig,jg,k]-C_in[ig,jg-1,k])/dy); end
            else; dy = HTM.get_dy_centers(grid,ig,jg); flux = -Kh*grid.face_area_y[ig,jg,k]*((C_in[ig,jg,k]-C_in[ig,jg-1,k])/dy); end
            w = isa(grid,CurvilinearGrid) ? grid.mask_v[ig,jg] : (grid.mask[ig,jg,k] & grid.mask[ig,jg-1,k])
            fy[ig,jg,k] = flux*w
        end
    end
    @inbounds for k in axes(C_out,3), j_phys in 1:ny, i_phys in 1:nx
        ig, jg = i_phys+ng, j_phys+ng
        C_out[ig,jg,k] = C_in[ig,jg,k] - (dt/grid.volume[ig,jg,k])*(fy[ig,jg+1,k]-fy[ig,jg,k])
    end
end

grid = initialize_curvilinear_grid(NC); hydro = create_hydrodynamic_data_from_file(NC); ds = NCDataset(NC)
function mkstate(ntr)
    names = ntuple(i->Symbol("T",i), ntr); st = initialize_state(grid, ds, names)
    update_hydrodynamics!(st, grid, ds, hydro, 10_886_400.0)
    nx,ny,nz=grid.nx,grid.ny,grid.nz; ng=grid.ng
    for nm in names, k in 1:nz, j in 1:ny, i in 1:nx
        st.tracers[nm][i+ng,j+ng,k] = Float32(exp(-((i-nx*0.4)^2+(j-ny*0.5)^2)/(2*30.0^2)))
    end
    st
end

# diffusion-only threaded tracer loop; ver = :old or :new
function run_diff!(st, ver)
    names = collect(keys(st.tracers)); ntr = length(names)
    nch = max(1, min(Threads.nthreads(), ntr)); HTM._ensure_flux_pools!(st, nch)
    Threads.@threads for cid in 1:nch
        fx = st.flux_x_pool[cid]; fy = st.flux_y_pool[cid]; ti = cid
        while ti <= ntr
            nm = names[ti]; Ci = st.tracers[nm]; Cm = st._buffer1[nm]
            if ver === :old
                old_diffuse_x!(Cm, Ci, st, grid, DT, KH, fx, DC); old_diffuse_y!(Ci, Cm, st, grid, DT, KH, fy, DC)
            else
                HTM.diffuse_x!(Cm, Ci, st, grid, DT, KH, DC); HTM.diffuse_y!(Ci, Cm, st, grid, DT, KH, DC)
            end
            ti += nch
        end
    end
end

# correctness: same input -> compare both single sweeps
function check(st)
    nm = first(keys(st.tracers)); Ci = copy(st.tracers[nm])
    fx = zeros(Float32, size(st.flux_x)); a = similar(st._buffer1[nm]); b = similar(st._buffer1[nm])
    old_diffuse_x!(a, Ci, st, grid, DT, KH, fx, DC)
    HTM.diffuse_x!(b, Ci, st, grid, DT, KH, DC)
    dx = maximum(abs.(Float64.(a) .- Float64.(b)))
    fy = zeros(Float32, size(st.flux_y)); old_diffuse_y!(a, Ci, st, grid, DT, KH, fy, DC); HTM.diffuse_y!(b, Ci, st, grid, DT, KH, DC)
    dy = maximum(abs.(Float64.(a) .- Float64.(b)))
    @printf("correctness max|old-new|:  x=%.3e  y=%.3e  (0 => bit-identical)\n", dx, dy)
end

function bench(st)
    run_diff!(st,:old); run_diff!(st,:new)  # warmup both
    bo=Inf; bn=Inf
    for _ in 1:8           # interleave old/new each batch -> thermal drift cancels
        to=@elapsed (for _ in 1:10; run_diff!(st,:old); end); bo=min(bo,to/10)
        tn=@elapsed (for _ in 1:10; run_diff!(st,:new); end); bn=min(bn,tn/10)
    end
    bo, bn
end

# full horizontal step (FFSL advection + diffusion); diffusion = old or new
function run_full!(st, ver)
    names = collect(keys(st.tracers)); ntr = length(names)
    nch = max(1, min(Threads.nthreads(), ntr)); HTM._ensure_flux_pools!(st, nch)
    HTM._compute_face_courant!(st, grid, DT, DC)
    Threads.@threads for cid in 1:nch
        fx = st.flux_x_pool[cid]; fy = st.flux_y_pool[cid]; ti = cid
        while ti <= ntr
            nm = names[ti]; Ci = st.tracers[nm]; Cm = st._buffer1[nm]
            HTM.advect_x_ffsl!(Cm, Ci, st, grid, DT, DC); HTM.advect_y_ffsl!(Ci, Cm, st, grid, DT, DC)
            if ver === :old
                old_diffuse_x!(Cm, Ci, st, grid, DT, KH, fx, DC); old_diffuse_y!(Ci, Cm, st, grid, DT, KH, fy, DC)
            else
                HTM.diffuse_x!(Cm, Ci, st, grid, DT, KH, DC); HTM.diffuse_y!(Ci, Cm, st, grid, DT, KH, DC)
            end
            ti += nch
        end
    end
end
function bench_full(st)
    run_full!(st,:old); run_full!(st,:new); bo=Inf; bn=Inf
    for _ in 1:8
        to=@elapsed (for _ in 1:10; run_full!(st,:old); end); bo=min(bo,to/10)
        tn=@elapsed (for _ in 1:10; run_full!(st,:new); end); bn=min(bn,tn/10)
    end
    bo, bn
end

for ntr in (8, 20)
    st = mkstate(ntr); ntr==8 && check(st)
    bo, bn = bench(st)
    @printf("ntr=%2d : diff OLD=%.1f  NEW=%.1f ms  | NEW/OLD=%.2f  (%+.0f%%)\n",
            ntr, bo*1e3, bn*1e3, bn/bo, 100*(bn/bo-1))
    fo, fn = bench_full(st)
    @printf("ntr=%2d : FULL OLD=%.1f  NEW=%.1f ms  | NEW/OLD=%.2f  (%+.0f%%)\n",
            ntr, fo*1e3, fn*1e3, fn/fo, 100*(fn/fo-1))
end
close(ds)
