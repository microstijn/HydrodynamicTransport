# src/OysterModule.jl

module OysterModule

export update_oysters!

using ..HydrodynamicTransport.ModelStructs

_f_temp(T) = exp(-0.006 * (T - 27.0)^2)

function _f_salinity(S)
    if S < 5.0; return 0.0;
    elseif S <= 12.0; return 0.0926 * (S - 0.0139);
    else; return 1.0;
    end
end

function _f_tss(TSS)
    if TSS < 4.0; return 0.1;
    elseif TSS <= 25.0; return 1.0;
    else; return 10.364 * log(TSS)^(-2.0477);
    end
end

function update_oysters!(model_state::State, grid::AbstractGrid, virtual_oysters::Vector{VirtualOyster}, dt_seconds::Float64, dissolved_tracer::Symbol, sorbed_tracer::Symbol)
    if isempty(virtual_oysters); return; end
    
    dt_days = dt_seconds / 86400.0

    for oyster in virtual_oysters
        i, j, k = oyster.i + grid.ng, oyster.j + grid.ng, oyster.k
        p = oyster.params
        
        T = model_state.temperature[i, j, k]
        S = model_state.salinity[i, j, k]
        TSS = model_state.tss[i, j, k]
        C_diss = model_state.tracers[dissolved_tracer][i, j, k]
        C_sorb = model_state.tracers[sorbed_tracer][i, j, k]

        scaling_factor = p.wdw^0.75
        fr_l_per_hr = 0.17 * scaling_factor * _f_temp(T) * _f_salinity(S) * _f_tss(TSS)
        fr_l_per_day = fr_l_per_hr * 24.0

        fpseudo = 0.0
        if TSS > p.tss_reject
            fpseudo = (TSS - p.tss_reject) / (p.tss_clog - p.tss_reject)
            fpseudo = clamp(fpseudo, 0.0, 1.0)
        end

        assimilated_free = fr_l_per_day * C_diss * p.ϵ_free
        ingested_sorbed = fr_l_per_day * C_sorb * (1 - fpseudo)
        assimilated_sorbed = ingested_sorbed * p.ϵ_sorbed
        total_uptake_rate = assimilated_free + assimilated_sorbed

        k_dep = p.kdep_20 * p.θ_dep^(T - 20.0)
        total_elimination = k_dep * oyster.state.c_oyster

        dCoyster_dt = (total_uptake_rate / p.wdw) - total_elimination
        
        new_c_oyster = oyster.state.c_oyster + dCoyster_dt * dt_days
        oyster.state.c_oyster = max(0.0, new_c_oyster)
    end
end

end # module OysterModule