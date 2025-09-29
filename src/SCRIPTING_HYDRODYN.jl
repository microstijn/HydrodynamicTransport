using Pkg
#Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HydrodynamicTransport
using UnicodePlots
using NCDatasets

run_all_tests();

run_integration_tests()

        # --- 1. Configuration (used by both tests) ---
        netcdf_filepath = "https://ns9081k.hyrax.sigma2.no/opendap/K160_bgc/Sim2/ocean_his_0001.nc"
        
        # This map is specific to the Norway ROMS file.
        # This file contains "u", "v", and "ocean_time", but not "temp" or "salt".
        variable_map = Dict(
            :u => "u",
            :v => "v",
            :time => "ocean_time"
        )
        hydro_data = HydrodynamicData(netcdf_filepath, variable_map)

        ds = NCDataset(netcdf_filepath)