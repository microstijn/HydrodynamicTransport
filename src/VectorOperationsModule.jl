# src/VectorOperationsModule.jl

module VectorOperationsModule

export rotate_velocities_to_geographic, rotate_velocities_to_grid! # Note the "!"

using ..HydrodynamicTransport.ModelStructs

"""
    rotate_velocities_to_geographic(grid::CurvilinearGrid, u_stag::AbstractArray, v_stag::AbstractArray)
... (this function is unchanged) ...
"""
function rotate_velocities_to_geographic(grid::CurvilinearGrid, u_stag::AbstractArray, v_stag::AbstractArray)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    u_east = zeros(Float64, nx, ny, nz); v_north = zeros(Float64, nx, ny, nz)
    u_rho = zeros(Float64, nx, ny, nz); v_rho = zeros(Float64, nx, ny, nz)

    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        u_rho[i_phys, j_phys, k] = 0.5 * (u_stag[i_glob, j_glob, k] + u_stag[i_glob+1, j_glob, k])
        v_rho[i_phys, j_phys, k] = 0.5 * (v_stag[i_glob, j_glob, k] + v_stag[i_glob, j_glob+1, k])
    end

    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        ur = u_rho[i_phys, j_phys, k]; vr = v_rho[i_phys, j_phys, k]
        ang = grid.angle[i_glob, j_glob]
        u_east[i_phys, j_phys, k] = ur * cos(ang) - vr * sin(ang)
        v_north[i_phys, j_phys, k] = vr * cos(ang) + ur * sin(ang)
    end
    return u_east, v_north
end


"""
    rotate_velocities_to_grid!(u_stag, v_stag, grid, u_east, v_north)

Rotates geographic velocities (East, North) at cell centers to grid-aligned velocities (u, v) 
and fills them IN-PLACE into the pre-allocated u_stag and v_stag arrays.
"""
function rotate_velocities_to_grid!(u_stag::AbstractArray, v_stag::AbstractArray, grid::CurvilinearGrid, u_east::AbstractArray, v_north::AbstractArray)
    ng = grid.ng
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    u_rho = zeros(Float64, nx, ny, nz)
    v_rho = zeros(Float64, nx, ny, nz)

    # 1. Apply the inverse rotation at each physical cell center
    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        ue = u_east[i_phys, j_phys, k]; vn = v_north[i_phys, j_phys, k]
        ang = grid.angle[i_glob, j_glob]
        u_rho[i_phys, j_phys, k] = ue * cos(-ang) - vn * sin(-ang)
        v_rho[i_phys, j_phys, k] = vn * cos(-ang) + ue * sin(-ang)
    end

    # 2. Interpolate from rho-points to staggered faces (physical domain only)
    # This now writes directly into the provided u_stag and v_stag arrays
    for k in 1:nz, j_phys in 1:ny, i_phys in 1:nx+1
        i_glob, j_glob = i_phys + ng, j_phys + ng
        if i_phys == 1
            u_stag[i_glob, j_glob, k] = u_rho[i_phys, j_phys, k]
        elseif i_phys == nx + 1
            u_stag[i_glob, j_glob, k] = u_rho[i_phys-1, j_phys, k]
        else
            u_stag[i_glob, j_glob, k] = 0.5 * (u_rho[i_phys-1, j_phys, k] + u_rho[i_phys, j_phys, k])
        end
    end
    
    for k in 1:nz, j_phys in 1:ny+1, i_phys in 1:nx
        i_glob, j_glob = i_phys + ng, j_phys + ng
        if j_phys == 1
            v_stag[i_glob, j_glob, k] = v_rho[i_phys, j_phys, k]
        elseif j_phys == ny + 1
            v_stag[i_glob, j_glob, k] = v_rho[i_phys, j_phys-1, k]
        else
            v_stag[i_glob, j_glob, k] = 0.5 * (v_rho[i_phys, j_phys-1, k] + v_rho[i_phys, j_phys, k])
        end
    end
    
    return nothing # The function modifies u_stag and v_stag in-place
end


end # module VectorOperationsModule