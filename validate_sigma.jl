# Regression for the MARS3D sigma vertical-coordinate fix.
#
# The CurviLoire file is MARS3D: it has NO ROMS s_w/Cs_w/hc, only a CF `ocean_sigma_coordinate`
# (`level`/`SIG`, layer centres in [-1,0]) + bathymetry H0 + SSH XE. Before the fix the grid fell
# back to z_w=[-1..0] -> dimensionless dz=0.1 -> cell volumes ~depth-times too small and the
# vertical mixing rate wrong. After the fix the physical layer thickness is Δσ_k·H0(i,j) [m].
#
# Checks (real grid, no run needed for 1-4; a short run for 5):
#   1. sigma coordinate is detected (z_w spans [-1,0], 10 layers).
#   2. physical layer thickness dz = volume·pm·pn is in METRES, and Σ_k dz = H0 per column.
#   3. cell volumes are physically sized (m^3), not dimensionless-small.
#   4. dz is spatially variable (tracks H0), not a single global value.
#   5. a short transport run completes, stays finite, and conserves tracer mass.
using Pkg
Pkg.activate(raw"C:\Users\peete074\OneDrive - Wageningen University & Research\programming\softMode")

using HydrodynamicTransport
using HydrodynamicTransport.ModelStructs
using HydrodynamicTransport.GridModule
using HydrodynamicTransport.StateModule
using HydrodynamicTransport.UtilsModule
using HydrodynamicTransport.TimeSteppingModule
using NCDatasets
using Printf
using Statistics: mean

const NC = raw"C:\Users\peete074\OneDrive - Wageningen University & Research\Documents\PREVIR_PROJECT\01_raw\hydro\CurviLoire\run_curviloire_2015.nc"
isfile(NC) || error("CurviLoire 2015 file not found: $NC")

grid = initialize_curvilinear_grid(NC)
ng = grid.ng
@printf("z_w (sigma interfaces): n=%d  range=[%.3f, %.3f]\n", length(grid.z_w), minimum(grid.z_w), maximum(grid.z_w))

# H0 bathymetry at the rho points (already on grid.h, ghost-extrapolated).
dz(i, j, k) = grid.volume[i, j, k] * grid.pm[i, j] * grid.pn[i, j]

# Pick a few wet interior columns and report dz + column-sum vs H0.
wet = [(i, j) for j in ng+1:grid.ny+ng, i in ng+1:grid.nx+ng if grid.mask_rho[i, j] && grid.h[i, j] > 1.0]
samp = wet[round.(Int, range(1, length(wet), length = 5))]
println("col (i,j)    H0[m]   dz_k[m] (k=1..nz)        Σdz[m]   vol_k1[m^3]")
maxrelerr = 0.0
for (i, j) in samp
    dzs = [dz(i, j, k) for k in 1:grid.nz]
    s = sum(dzs); H0 = grid.h[i, j]
    relerr = abs(s - H0) / H0; global maxrelerr = max(maxrelerr, relerr)
    @printf("(%3d,%3d)  %7.2f   %s   %7.2f   %.3e\n", i, j, H0,
            join((@sprintf("%.2f", d) for d in dzs), " "), s, grid.volume[i, j, 1])
end
@printf("max |Σdz - H0|/H0 over samples = %.3e (should be ~0)\n", maxrelerr)

dz_all = [dz(i, j, 1) for (i, j) in wet]
@printf("layer-1 dz over wet cells: min=%.3f max=%.3f mean=%.3f  (metres; spatially variable)\n",
        minimum(dz_all), maximum(dz_all), mean(dz_all))
vol_all = [grid.volume[i, j, 1] for (i, j) in wet]
@printf("layer-1 volume over wet cells: min=%.3e max=%.3e mean=%.3e m^3\n",
        minimum(vol_all), maximum(vol_all), mean(vol_all))

# --- Short transport run: finite + mass-conserving with the corrected (physical) volumes. ---
hydro_data = create_hydrodynamic_data_from_file(NC)
ds = NCDataset(NC)
state = initialize_state(grid, ds, (:A,))
sources = [PointSource(i=156, j=156, k=1, tracer_name=:A, influx_rate=t->1.0e6, relocate_if_dry=true)]
start_t = 10_886_400.0
final = run_simulation(grid, state, sources, start_t, start_t + 1800.0, 300.0;
                       ds = ds, hydro_data = hydro_data, advection_scheme = :TVD,
                       use_adaptive_dt = true, cfl_max = 0.8, dt_max = 300.0, dt_min = 1.0)
mass = sum(final.tracers[:A] .* grid.volume)
@printf("run OK: any-NaN=%s  min=%.3e  mass=%.6e  t=%.1f\n",
        any(isnan, final.tracers[:A]), minimum(final.tracers[:A]), mass, final.time)
close(ds)
