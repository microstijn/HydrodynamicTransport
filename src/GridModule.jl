# src/GridModule.jl

module GridModule

export initialize_cartesian_grid, initialize_curvilinear_grid

using ..HydrodynamicTransport.ModelStructs
using StaticArrays
using NCDatasets
using Statistics: mean

# The Cartesian grid initializer remains unchanged.
function initialize_cartesian_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64; ng::Int=2)::CartesianGrid
    dx = Lx / nx; dy = Ly / ny; dz = Lz / nz
    nx_tot, ny_tot = nx + 2*ng, ny + 2*ng
    x = zeros(nx_tot, ny_tot, nz); y = zeros(nx_tot, ny_tot, nz); z = zeros(nx_tot, ny_tot, nz)
    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        x[i_glob, j_glob, k] = (i_phys - 0.5) * dx
        y[i_glob, j_glob, k] = (j_phys - 0.5) * dy
        z[i_glob, j_glob, k] = (k - 0.5) * dz
    end
    grid = CartesianGrid(
        ng, SVector(nx, ny, nz), x, y, z,
        fill(dx * dy * dz, (nx_tot, ny_tot, nz)),
        fill(dy * dz, (nx_tot + 1, ny_tot, nz)),
        fill(dx * dz, (nx_tot, ny_tot + 1, nz)),
        fill(dx * dy, (nx_tot, ny_tot, nz + 1)),
        trues(nx_tot, ny_tot, nz)
    )
    return grid
end

# --- Internal function to auto-detect grid variable and dimension names ---
function _autodetect_grid_vars_and_dims(ds::NCDataset)
    var_map = Dict{Symbol, String}()
    
    search_patterns = [
        :ni_rho => [("ni", :dim), ("xi_rho", :dim)],
        :nj_rho => [("nj", :dim), ("eta_rho", :dim)],
        :ni_u => [("ni_u", :dim), ("xi_u", :dim)],
        :nj_u => [("nj_u", :dim), ("eta_u", :dim)],
        :ni_v => [("ni_v", :dim), ("xi_v", :dim)],
        :nj_v => [("nj_v", :dim), ("eta_v", :dim)],
        :nz => [("level", :dim), ("s_rho", :dim)],
        :nz_w => [("level_w", :dim), ("s_w", :dim)],
        :lon_rho => [("longitude", :var), ("lon_rho", :var)],
        :lat_rho => [("latitude", :var), ("lat_rho", :var)],
        :lon_u => [("longitude_u", :var), ("lon_u", :var)],
        :lat_u => [("latitude_u", :var), ("lat_u", :var)],
        :lon_v => [("longitude_v", :var), ("lon_v", :var)],
        :lat_v => [("latitude_v", :var), ("lat_v", :var)],
        :h => [("H0", :var), ("h", :var)],
        :angle => [("angle", :var)],
        :mask_rho => [("mask_rho", :var)],
        :mask_u => [("mask_u", :var)],
        :mask_v => [("mask_v", :var)],
        :s_w => [("s_w", :var)],
        :Cs_w => [("Cs_w", :var)],
        :hc => [("hc", :var)],
    ]

    if haskey(ds, "pm") && haskey(ds, "pn"); var_map[:pm] = "pm"; var_map[:pn] = "pn"
    elseif haskey(ds, "dx") && haskey(ds, "dy"); var_map[:pm] = "dx"; var_map[:pn] = "dy"; var_map[:is_inverse_metric] = "true"
    else; error("Could not find grid metric variables (pm/pn or dx/dy) in the NetCDF file."); end

    for (target_symbol, patterns) in search_patterns
        for (name, type) in patterns
            collection = (type == :var) ? keys(ds) : keys(ds.dim)
            if name in collection; var_map[target_symbol] = name; break; end
        end
    end
    
    return var_map
end


# --- Refactored Curvilinear Grid Initializer with Auto-Detection ---
function initialize_curvilinear_grid(netcdf_filepath::String; ng::Int=2)::CurvilinearGrid
    ds = NCDataset(netcdf_filepath)
    grid_vars = _autodetect_grid_vars_and_dims(ds)
    nx_rho, ny_rho = ds.dim[grid_vars[:ni_rho]], ds.dim[grid_vars[:nj_rho]]; nx_u, ny_u = ds.dim[grid_vars[:ni_u]], ds.dim[grid_vars[:nj_u]]; nx_v, ny_v = ds.dim[grid_vars[:ni_v]], ds.dim[grid_vars[:nj_v]]; nz = ds.dim[grid_vars[:nz]]
    lon_rho_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); lat_rho_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); pm_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); pn_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); angle_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); h_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng); mask_rho_full= ones(Bool, nx_rho + 2*ng, ny_rho + 2*ng); lon_u_full = zeros(Float64, nx_u + 2*ng, ny_u + 2*ng); lat_u_full = zeros(Float64, nx_u + 2*ng, ny_u + 2*ng); mask_u_full = ones(Bool, nx_u + 2*ng, ny_u + 2*ng); lon_v_full = zeros(Float64, nx_v + 2*ng, ny_v + 2*ng); lat_v_full = zeros(Float64, nx_v + 2*ng, ny_v + 2*ng); mask_v_full = ones(Bool, nx_v + 2*ng, ny_v + 2*ng)
    rho_interior(A) = view(A, ng+1:nx_rho+ng, ng+1:ny_rho+ng); u_interior(A) = view(A, ng+1:nx_u+ng, ng+1:ny_u+ng); v_interior(A) = view(A, ng+1:nx_v+ng, ng+1:ny_v+ng)
    
    rho_interior(lon_rho_full) .= coalesce.(ds[grid_vars[:lon_rho]][:,:], 0.0); rho_interior(lat_rho_full) .= coalesce.(ds[grid_vars[:lat_rho]][:,:], 0.0)
    rho_interior(h_full) .= coalesce.(ds[grid_vars[:h]][:,:], 0.0)

    # --- FIX: Intelligently create the mask based on what's available ---
    if haskey(grid_vars, :mask_rho)
        rho_interior(mask_rho_full) .= (coalesce.(ds[grid_vars[:mask_rho]][:,:], 0) .== 1)
    else # Infer mask from bathymetry (H0 > 0 is water)
        rho_interior(mask_rho_full) .= (rho_interior(h_full) .> 0.0)
    end
    
    if haskey(grid_vars, :is_inverse_metric); rho_interior(pm_full) .= 1 ./ coalesce.(ds[grid_vars[:pm]][:,:], 1.0); rho_interior(pn_full) .= 1 ./ coalesce.(ds[grid_vars[:pn]][:,:], 1.0); else; rho_interior(pm_full) .= coalesce.(ds[grid_vars[:pm]][:,:], 0.0); rho_interior(pn_full) .= coalesce.(ds[grid_vars[:pn]][:,:], 0.0); end
    if haskey(grid_vars, :angle); rho_interior(angle_full) .= coalesce.(ds[grid_vars[:angle]][:,:], 0.0); end
    if haskey(grid_vars, :lon_u); u_interior(lon_u_full) .= coalesce.(ds[grid_vars[:lon_u]][:,:], 0.0); end; if haskey(grid_vars, :lat_u); u_interior(lat_u_full) .= coalesce.(ds[grid_vars[:lat_u]][:,:], 0.0); end
    if haskey(grid_vars, :mask_u); u_interior(mask_u_full) .= (coalesce.(ds[grid_vars[:mask_u]][:,:], 0) .== 1); else; u_interior(mask_u_full) .= (view(h_full, ng+1:nx_u+ng, ng+1:ny_u+ng) .> 0.0) .& (view(h_full, ng+2:nx_u+ng+1, ng+1:ny_u+ng) .> 0.0); end
    if haskey(grid_vars, :lon_v); v_interior(lon_v_full) .= coalesce.(ds[grid_vars[:lon_v]][:,:], 0.0); end; if haskey(grid_vars, :lat_v); v_interior(lat_v_full) .= coalesce.(ds[grid_vars[:lat_v]][:,:], 0.0); end
    if haskey(grid_vars, :mask_v); v_interior(mask_v_full) .= (coalesce.(ds[grid_vars[:mask_v]][:,:], 0) .== 1); else; v_interior(mask_v_full) .= (view(h_full, ng+1:nx_v+ng, ng+1:ny_v+ng) .> 0.0) .& (view(h_full, ng+1:nx_v+ng, ng+2:ny_v+ng+1) .> 0.0); end

    z_w = if haskey(grid_vars, :s_w) && haskey(grid_vars, :Cs_w) && haskey(grid_vars, :hc); s_w = ds[grid_vars[:s_w]][:]; Cs_w = ds[grid_vars[:Cs_w]][:]; hc = ds[grid_vars[:hc]][:]; h_phys = ds[grid_vars[:h]][:,:]; h_avg = mean(coalesce.(h_phys[h_phys .> 0], 0.0)); h_avg .* (hc .* s_w .+ h_avg .* Cs_w) ./ (hc .+ h_avg); else; collect(range(-1.0, 0.0, length=nz+1)); end
    close(ds)
    extrapolate!(A) = begin; nx_p, ny_p = size(A) .- 2*ng; for j in ng+1:ny_p+ng; A[1:ng, j] .= A[ng+1, j]; A[nx_p+ng+1:end, j] .= A[nx_p+ng, j]; end; for i in 1:size(A, 1); A[i, 1:ng] .= A[i, ng+1]; A[i, ny_p+ng+1:end] .= A[i, ny_p+ng]; end; end
    for arr in [lon_rho_full, lat_rho_full, pm_full, pn_full, angle_full, h_full, mask_rho_full, lon_u_full, lat_u_full, mask_u_full, lon_v_full, lat_v_full, mask_v_full]; extrapolate!(arr); end
    nx_tot, ny_tot = nx_rho + 2*ng, ny_rho + 2*ng; face_area_x = zeros(nx_tot + 1, ny_tot, nz); face_area_y = zeros(nx_tot, ny_tot + 1, nz); volume = zeros(nx_tot, ny_tot, nz); dz = z_w[2:end] - z_w[1:end-1]
    for k in 1:nz; for j in 1:ny_tot, i in 1:nx_tot; volume[i,j,k] = (1/pm_full[i,j]) * (1/pn_full[i,j]) * abs(dz[k]); end; for j in 1:ny_tot, i in 1:nx_tot+1; dy_local = (i > 1 && i <= nx_tot) ? 0.5 * (1/pn_full[i-1,j] + 1/pn_full[i,j]) : 1/pn_full[min(i, nx_tot), j]; face_area_x[i,j,k] = dy_local * abs(dz[k]); end; for j in 1:ny_tot+1, i in 1:nx_tot; dx_local = (j > 1 && j <= ny_tot) ? 0.5 * (1/pm_full[i,j-1] + 1/pm_full[i,j]) : 1/pm_full[i, min(j, ny_tot)]; face_area_y[i,j,k] = dx_local * abs(dz[k]); end; end
    return CurvilinearGrid(ng, nx_rho, ny_rho, nz, lon_rho_full, lat_rho_full, lon_u_full, lat_u_full, lon_v_full, lat_v_full, z_w, pm_full, pn_full, angle_full, h_full, mask_rho_full, mask_u_full, mask_v_full, face_area_x, face_area_y, volume)
end

end # module GridModule

