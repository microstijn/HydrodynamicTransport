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


# --- Refactored Curvilinear Grid Initializer ---
function initialize_curvilinear_grid(netcdf_filepath::String; ng::Int=2)::CurvilinearGrid
    ds = NCDataset(netcdf_filepath)

    # --- 1. Read Physical Dimensions ---
    nx_rho, ny_rho = ds.dim["xi_rho"], ds.dim["eta_rho"]
    nx_u,   ny_u   = ds.dim["xi_u"],   ds.dim["eta_u"]
    nx_v,   ny_v   = ds.dim["xi_v"],   ds.dim["eta_v"]
    nz = ds.dim["s_rho"]

    # --- 2. Allocate Full-Sized Arrays (including ghost cells) ---
    # Rho-point arrays
    lon_rho_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    lat_rho_full = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    pm_full      = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    pn_full      = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    angle_full   = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    h_full       = zeros(Float64, nx_rho + 2*ng, ny_rho + 2*ng)
    mask_rho_full= zeros(Bool,    nx_rho + 2*ng, ny_rho + 2*ng)
    
    # U-point arrays
    lon_u_full   = zeros(Float64, nx_u + 2*ng, ny_u + 2*ng)
    lat_u_full   = zeros(Float64, nx_u + 2*ng, ny_u + 2*ng)
    mask_u_full  = zeros(Bool,    nx_u + 2*ng, ny_u + 2*ng)
    
    # V-point arrays
    lon_v_full   = zeros(Float64, nx_v + 2*ng, ny_v + 2*ng)
    lat_v_full   = zeros(Float64, nx_v + 2*ng, ny_v + 2*ng)
    mask_v_full  = zeros(Bool,    nx_v + 2*ng, ny_v + 2*ng)

    # --- 3. Create Views into the Physical Interior ---
    rho_interior(A) = view(A, ng+1:nx_rho+ng, ng+1:ny_rho+ng)
    u_interior(A)   = view(A, ng+1:nx_u+ng,   ng+1:ny_u+ng)
    v_interior(A)   = view(A, ng+1:nx_v+ng,   ng+1:ny_v+ng)

    # --- 4. Read Data Directly into Interior Views (Corrected Syntax) ---
    # The .= syntax reads the data into a temporary array and then broadcasts/copies it into the view.
    rho_interior(lon_rho_full) .= ds["lon_rho"][:,:]
    rho_interior(lat_rho_full) .= ds["lat_rho"][:,:]
    rho_interior(pm_full)      .= ds["pm"][:,:]
    rho_interior(pn_full)      .= ds["pn"][:,:]
    rho_interior(angle_full)   .= ds["angle"][:,:]
    rho_interior(h_full)       .= ds["h"][:,:]
    rho_interior(mask_rho_full) .= (ds["mask_rho"][:,:] .== 1)

    u_interior(lon_u_full) .= ds["lon_u"][:,:]
    u_interior(lat_u_full) .= ds["lat_u"][:,:]
    u_interior(mask_u_full) .= (ds["mask_u"][:,:] .== 1)

    v_interior(lon_v_full) .= ds["lon_v"][:,:]
    v_interior(lat_v_full) .= ds["lat_v"][:,:]
    v_interior(mask_v_full) .= (ds["mask_v"][:,:] .== 1)

    # --- FIX: Compute z_w dynamically from S-coordinate variables ---
    # 1. Read S-coordinate parameters from the file
    s_w = ds["s_w"][:]
    Cs_w = ds["Cs_w"][:]
    hc = ds["hc"][:]
    h_phys = ds["h"][:,:]
    
    # 2. Calculate average depth (ignoring land points where h might be small/zero)
    h_avg = mean(h_phys[ds["mask_rho"][:,:] .== 1])
    
    # 3. Apply the ROMS stretching formula (assuming Vtransform = 2) to get a representative 1D vertical grid
    # z_w(s) = h_avg * (hc * s + h_avg * Cs_w(s)) / (hc + h_avg)
    z_w = h_avg .* (hc .* s_w .+ h_avg .* Cs_w) ./ (hc .+ h_avg)
    
    close(ds)

    # --- 5. Extrapolate Grid Metrics into Ghost Cells (Two-Pass for Corners) ---
    # Helper for 2D arrays
    extrapolate!(A) = begin
        nx_phys, ny_phys = size(A) .- 2*ng
        # Pass 1: East-West
        for j in ng+1:ny_phys+ng
            A[1:ng, j] .= A[ng+1, j]       # West
            A[nx_phys+ng+1:nx_phys+2*ng, j] .= A[nx_phys+ng, j] # East
        end
        # Pass 2: North-South (covers corners)
        for i in 1:nx_phys+2*ng
            A[i, 1:ng] .= A[i, ng+1]       # South
            A[i, ny_phys+ng+1:ny_phys+2*ng] .= A[i, ny_phys+ng] # North
        end
    end
    
    # Extrapolate all metric and coordinate arrays
    for arr in [lon_rho_full, lat_rho_full, pm_full, pn_full, angle_full, h_full, mask_rho_full,
                lon_u_full, lat_u_full, mask_u_full, lon_v_full, lat_v_full, mask_v_full]
        extrapolate!(arr)
    end
    
    # --- 6. Calculate 3D Grid Properties on the Full Grid ---
    nx_tot, ny_tot = nx_rho + 2*ng, ny_rho + 2*ng
    face_area_x = zeros(nx_tot + 1, ny_tot, nz)
    face_area_y = zeros(nx_tot, ny_tot + 1, nz)
    volume = zeros(nx_tot, ny_tot, nz)

    dz = z_w[2:end] - z_w[1:end-1]

    for k in 1:nz
        for j in 1:ny_tot, i in 1:nx_tot
            volume[i,j,k] = (1/pm_full[i,j]) * (1/pn_full[i,j]) * abs(dz[k])
        end
        # Note: Face area calculations are approximations at cell faces
        for j in 1:ny_tot, i in 1:nx_tot+1
            local_dy = (i > 1 && i <= nx_tot) ? 0.5 * (1/pn_full[i-1,j] + 1/pn_full[i,j]) : 1/pn_full[min(i, nx_tot), j]
            face_area_x[i,j,k] = local_dy * abs(dz[k])
        end
        for j in 1:ny_tot+1, i in 1:nx_tot
            local_dx = (j > 1 && j <= ny_tot) ? 0.5 * (1/pm_full[i,j-1] + 1/pm_full[i,j]) : 1/pm_full[i, min(j, ny_tot)]
            face_area_y[i,j,k] = local_dx * abs(dz[k])
        end
    end

    # --- 7. Construct and Return the Grid Struct ---
    return CurvilinearGrid(ng, nx_rho, ny_rho, nz, 
                           lon_rho_full, lat_rho_full, lon_u_full, lat_u_full, lon_v_full, lat_v_full, 
                           z_w, pm_full, pn_full, angle_full, h_full, 
                           mask_rho_full, mask_u_full, mask_v_full, 
                           face_area_x, face_area_y, volume)
end

end # module GridModule

