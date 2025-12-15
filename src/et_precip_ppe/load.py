import warnings
import numpy as np
import pandas as pd
import xarray as xr

### Constants ###

# The CMIP6 models with the necessary abrupt-CO2 experiments
# that were used in this manuscript
CMIP_MODEL = [
    "CESM2",
    "CanESM5",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MRI-ESM2-0",
]

# Unit conversions, to be multiplied
UNIT_CONV = {
    "water flux [W/m2 -> mm/day]": 60*60*24/2.26e6,
    "water flux [m/s -> mm/day]": 86400,
    "specific humidity [kg/kg -> g/kg]": 1000,
}

"""
The following variable dictionaries create a common set of 
variable names for CMIP, CESM, and HadCM. Some common variable
names are not actually output from the models and need to be
derived from other output variables. These cases are dealt with
in the get_cmip, get_cesm, and get_hadcm functions. In the mapping
dictionaries, the variables labeled "not available" either are not
available or have not been processed because I do not need them.
"""

# Define the common variable names
VAR_DESCR = {
    "T":        "3d air temperature",
    "T_S":      "near-surface air temperature",
    "RH":       "3d relative humidity",
    "RH_S":     "near-surface relative humidity",
    "q":        "3d specific humidity",
    "q_S":      "near-surface specific humidity",
    "ET":       "evapotranspiration",
    "P":        "precipitation",
    "PmE":      "precipitation minus evapotranspiration",
    "LH":       "latent heat flux",
    "SH":       "sensible heat flux",
    "Ps":       "surface pressure",
    "Q_w":      "total precipitable water",
    "WP":       "total grid-box cloud water path (liquid and ice)",
    "LWCF":     "longwave cloud forcing",
    "SWCF":     "shortwave cloud forcing",
    "Rnet_TOA": "net radiative flux at the TOA",
    "albedo_S": "surface albedo",
    "U_10m":    "10m wind speed",
    "CLDLOW":   "low cloud fraction",
    "FSDS":     "downwelling surface shortwave flux",
    "FSNS":     "net surface shortwave flux",
    "FSNSC":    "clearsky net surface shortwave flux",
    "FSDSC":    "clearsky downwelling surface shortwave flux",
}

# Map the CMIP variable names to the common variable names
CMIP_VAR_DICT = {
    "Amon": {
        "T": "not available",
        "T_S": "tas",
        "RH": "not available",
        "RH_S": "hurs",
        "q": "not available",
        "q_S": "huss",
        "ET": "evspsbl",
        "P": "pr",
        "PmE": "derived",
        "LH": "not available",
        "SH": "not available",
        "Ps": "not available",
        "Q_w": "prw",
        "WP": "not available",
        "LWCF": "not available",
        "SWCF": "not available",
        "Rnet_TOA": "not available",
        "albedo_S": "not available",
        "U_10m": "not available",
    },
    "Lmon": { }
}

# Map the CESM variable names to the common variable names
CESM_VAR_DICT = {
    "T": "T",
    "T_S": "TREFHT",
    "RH": "RELHUM",
    "RH_S": "derived",
    "q": "Q",
    "q_S": "QREFHT",
    "ET": "derived",
    "P": "calculated_PRECT",
    "PmE": "derived",
    "LH": "LHFLX",
    "SH": "SHFLX",
    "Ps": "PS",
    "Q_w": "TMQ",
    "WP": "TGCLDCWP",
    "LWCF": "LWCF",
    "SWCF": "SWCF",
    "Rnet_TOA": "derived",
    "albedo_S": "derived",
    "U_10m": "U10",
    "CLDLOW": "CLDLOW",
    "FSDS": "FSDS",
    "FSNS": "FSNS",
    "FSNSC": "FSNSC",
    "FSDSC": "FSDSC",
}

# Map the HadCM variable names to the common variable names
HADCM_VAR_DICT = {
    "T": "not available",
    "T_S": "air_temperature",
    "RH": "not available",
    "RH_S": "relative_humidity",
    "q": "not available",
    "q_S": "specific_humidity",
    "ET": "surface_upward_water_flux",
    "P": "precipitation_flux",
    "PmE": "derived",
    "LH": "surface_upward_latent_heat_flux",
    "SH": "surface_upward_sensible_heat_flux",
    "Ps": "not available",
    "Q_w": "not available",
    "WP": "not available",
    "LWCF": "derived",
    "SWCF": "derived",
    "Rnet_TOA": "derived",
    "albedo_S": "derived",
    "U_10m": "wind_speed",
}


### Helper functions ###


def get_cesm_crosswalk():
    cesm_ppe_crosswalk = pd.read_csv("/glade/u/home/czarakas/coupled_PPE/code/02_set_up_ensemble/CLM5PPE_coupledPPE_crosswalk.csv")

    cesm_ppe_crosswalk = cesm_ppe_crosswalk.sort_values("key_coupledPPE")

    cesm_ppe_crosswalk["description"] = cesm_ppe_crosswalk["param"] + ", " + cesm_ppe_crosswalk["minmax"]
    cesm_ppe_crosswalk = cesm_ppe_crosswalk.drop(index=0)
    cesm_ppe_crosswalk = cesm_ppe_crosswalk.replace({"parameter domain": {np.nan: "Boundary layer"}})

    return cesm_ppe_crosswalk


def get_processed_type_labels(proc_type):
    """
    Gets the subdirectory name and file descriptor for the specified type of processed output
    """
    subdir_name = None
    proc_descr = None
    if proc_type == "annual":
        subdir_name = "annual_timeseries"
        proc_descr = "ann_mean_ts"
    if proc_type == "month_mean":
        subdir_name = "climatology"
        proc_descr = "climatology"
    if proc_type == "time_mean":
        subdir_name = "time_averages"
        proc_descr = "time_mean"
    assert subdir_name is not None, "invalid type of processed output"
    return subdir_name, proc_descr


def load_cmip_da(model, experiment, component, variable, subdir_name, proc_descr):
    path = f"/glade/work/bbuchovecky/CMIP_analysis/{model}/{experiment}/{subdir_name}/"
    filename = f"{experiment}.{model}.{component}.{proc_descr}.{variable}.nc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path+filename)
    return ds


def get_cmip_abruptco2(model, component, variable, subdir_name, proc_descr):
    # Experiment names
    experiment = ["abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2"]
    exp_dim = np.arange(len(experiment))

    # Load all experiments and add to a list
    ds_exp = []
    for exp in experiment:
        ds = load_cmip_da(model, exp, component, variable, subdir_name, proc_descr)
        ds_exp.append(ds)
    
    # Combine the list of experiments into a single DataArray with an experiment dimension
    ds_exp = xr.combine_nested(ds_exp, concat_dim="experiment", combine_attrs="override")
    ds_exp = ds_exp.assign_coords({"experiment": exp_dim})

    return ds_exp


def load_cesm_da(variable, component, subdir_name, proc_descr, cam_grid):
    inpath = f"/glade/work/bbuchovecky/PPE_analysis/{subdir_name}"

    # Load ensemble and standard members
    ppe = xr.open_dataset(f"{inpath}/COUP_PPE.{component}.h0.{proc_descr}.{variable}.nc")[variable]
    std = xr.open_dataset(f"{inpath}/COUP_REF.{component}.h0.{proc_descr}.{variable}.nc")[variable]
        
    # Eliminate the degenerate dimension
    if ("member" in std.dims) and (std.member.size == 1):
        std = std.isel(member=0)
    
    # Reindex lat/lon coordinates to the CAM grid for CLM variables
    if component == "clm2":
        for da in (ppe, std):
            da = da.reindex(lat=cam_grid.lat, lon=cam_grid.lon, method="nearest", tolerance=0.05)
    
    return ppe, std


def cmip_model_dict_to_da_dict(cmip_dict):
    """
    Takes a dictionary of CMIP DataArrays
    and collapses the model dimension into
    a single DataArray. See the dictionary
    structure change below:
        in_dict[domain][region][model][variable]
        =>
        out_dict[domain][region][variable]
    where domain is [lnd, ocn, all] and region
    is the zonal band which has been averaged.

    Parameters:
    -----------
    cmip_dict : dict
        Dictionary of CMIP DataArrays with the
        structure [domain][region][model][variable].
    
    Returns:
    --------
    cmip_da_dict : dict
        Dictionary of CMIP DataArrays with the
        structure [domain][region][variable] and
        a model dimension in the DataArray.
    """
    cmip_da_dict = {}

    for dom in cmip_dict.keys():
        cmip_da_dict[dom] = {}

        for reg in cmip_dict[dom].keys():
            cmip_da_dict[dom][reg] = {}
            placeholder_model = list(cmip_dict[dom][reg].keys())[0]

            for var in cmip_dict[dom][reg][placeholder_model].keys():
                cmip_da_dict[dom][reg][var] = []

                for model in cmip_dict[dom][reg].keys():
                    cmip_da_dict[dom][reg][var].append(cmip_dict[dom][reg][model][var].reset_coords(drop=True))

                cmip_da_dict[dom][reg][var] = xr.combine_nested(cmip_da_dict[dom][reg][var], "model")

    return cmip_da_dict


### Functions for loading processed model output ###


def get_cmip(varlist, proc_type):
    """
    Parameters:
    -----------
    varlist : list
        List of variables
    proc_type : string
        Type of processed output ["annual", "month_mean", "time_mean"]

    Returns:
    --------
    co2 : dict
        Dictionary with structure [model][variable] of processed CMIP6 abrupt-CO2 output
    pi : dict
        Dictionary with structure [model][variable] of processed CMIP6 piControl output
    weights : dict
        Dictionary with structure [model] of CMIP6 gridcell weights
    """
    # Get the subdirectory name and file descriptor for the specified type of processed output
    subdir_name, proc_descr = get_processed_type_labels(proc_type)

    # Initialize data dictionaries
    co2 = {}
    pi = {}
    weights = {}

    print(f"CMIP6 {proc_type}:")
    for model in CMIP_MODEL:
        print("  "+model, end=" -- ")

        # Initialize a dictionary for the model
        co2[model] = {}
        pi[model] = {}

        # Load array of gridcell weights
        ds = xr.open_dataset(f"/glade/work/bbuchovecky/CMIP_analysis/{model}/weights/{model}_weights_atmgrid.nc")
        weights[model] = ds.copy(deep=True)

        for variable in varlist:
            print(variable, end=", ")

            # Get the correct model component (atmosphere or land)
            if variable in CMIP_VAR_DICT["Amon"]:
                component = "Amon"
            if variable in CMIP_VAR_DICT["Lmon"]:
                component = "Lmon"
            
            # Get the CMIP variable name
            cmip_variable = CMIP_VAR_DICT[component][variable]

            # Compute P-E
            if variable == "PmE":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pr_co2 = get_cmip_abruptco2(model, component, "pr", subdir_name, proc_descr)["pr"]
                    pr_pi = load_cmip_da(model, "piControl", component, "pr", subdir_name, proc_descr)["pr"]

                    evspsbl_co2 = get_cmip_abruptco2(model, component, "evspsbl", subdir_name, proc_descr)["evspsbl"]
                    evspsbl_pi = load_cmip_da(model, "piControl", component, "evspsbl", subdir_name, proc_descr)["evspsbl"]

                co2[model][variable] = pr_co2 - evspsbl_co2
                pi[model][variable] = pr_pi - evspsbl_pi

            else:
                # Load abrupt-CO2 experiments
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    da = get_cmip_abruptco2(model, component, cmip_variable, subdir_name, proc_descr)[cmip_variable]
                co2[model][variable] = da.copy(deep=True)

                # Load piControl experiment
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    da = load_cmip_da(model, "piControl", component, cmip_variable, subdir_name, proc_descr)[cmip_variable]
                pi[model][variable] = da.copy(deep=True)

            # Unit conversion for ET and P
            if variable in ("ET", "P", "PmE"):
                co2[model][variable] = co2[model][variable] * UNIT_CONV["water flux [m/s -> mm/day]"]
                pi[model][variable] = pi[model][variable] * UNIT_CONV["water flux [m/s -> mm/day]"]
            
            # Unit conversion for q_S
            if variable == "q_S":
                co2[model][variable] = co2[model][variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]
                pi[model][variable] = pi[model][variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]

            co2[model][variable] = co2[model][variable].rename(variable)
            pi[model][variable] = pi[model][variable].rename(variable)
        print()
    print()

    return co2, pi, weights


def get_cesm(varlist, proc_type, cam_grid=None):
    """
    Parameters:
    -----------
    varlist : list
        List of variables
    proc_type : string
        Type of processed output ["annual", "month_mean", "time_mean"]

    Returns:
    --------
    ppe : dict
        Dictionary with structure [variable] of processed CESM2 PPE output
    std : dict
        Dictionary with structure [variable] of processed CESM2 standard simulation output
    """
    # Source directory
    inpath = "/glade/work/bbuchovecky/PPE_analysis"

    # Get the subdirectory name and file descriptor for the specified type of processed output
    subdir_name, proc_descr = get_processed_type_labels(proc_type)

    # Load the gridcell weights
    weights = xr.open_dataset(f"{inpath}/weights/COUP_PPE_weights_atmgrid.nc")

    # Use the gridcell weights as a reference CAM grid if not provided
    if cam_grid is None:
        cam_grid = weights

    # Initialize data dictionaries
    ppe = {}
    std = {}

    print(f"CESM2 {proc_type}:")
    for variable in varlist:
        print(f"  {variable}")

        # Get the CESM variable name
        cesm_variable = CESM_VAR_DICT[variable]

        # Get the correct model component
        if cesm_variable == "RH2M":
            component = "clm2"
        if cesm_variable != "RH2M":
            component = "cam"

        # Check if the variable needs to be calculated

        if variable == "q_S":
            ppe[variable], std[variable] = load_cesm_da(cesm_variable, "cam", subdir_name, proc_descr, cam_grid)
            ppe[variable] = ppe[variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]
            std[variable] = std[variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]

        # Convert the latent heat flux into water units for evaporation
        elif variable == "ET":
            lhflx_ppe, lhflx_std = load_cesm_da("LHFLX", "cam", subdir_name, proc_descr, cam_grid)
            ppe[variable] = lhflx_ppe * UNIT_CONV["water flux [W/m2 -> mm/day]"]
            std[variable] = lhflx_std * UNIT_CONV["water flux [W/m2 -> mm/day]"]

        # Compute P-E
        elif variable == "PmE":
            prect_ppe, prect_std = load_cesm_da("calculated_PRECT", "cam", subdir_name, proc_descr, cam_grid)
        
            lhflx_ppe, lhflx_std = load_cesm_da("LHFLX", "cam", subdir_name, proc_descr, cam_grid)
            et_ppe = lhflx_ppe * UNIT_CONV["water flux [W/m2 -> mm/day]"]
            et_std = lhflx_std * UNIT_CONV["water flux [W/m2 -> mm/day]"]

            ppe[variable] = prect_ppe - et_ppe
            std[variable] = prect_std - et_std

        # Use calculated_RHREFHT
        elif variable == "RH_S":
            ppe[variable], std[variable] = load_cesm_da("calculated_RHREFHT", "cam", subdir_name, proc_descr, cam_grid)

        # Compute surface albedo from the downwelling surface shortwave flux and the net surface shortwave flux
        elif variable == "albedo_S":
            fsds_ppe, fsds_std = load_cesm_da("FSDS", "cam", subdir_name, proc_descr, cam_grid)
            fsns_ppe, fsns_std = load_cesm_da("FSNS", "cam", subdir_name, proc_descr, cam_grid)

            ppe[variable] = (fsds_ppe - fsns_ppe) / fsds_ppe
            std[variable] = (fsds_std - fsns_std) / fsds_std

        # Compute the net TOA radiative flux from the net TOA shortwave and longwave fluxes
        elif variable == "Rnet_TOA":
            fsnt_ppe, fsnt_std = load_cesm_da("FSNT", "cam", subdir_name, proc_descr, cam_grid)
            flnt_ppe, flnt_std = load_cesm_da("FLNT", "cam", subdir_name, proc_descr, cam_grid)

            ppe[variable] = fsnt_ppe - flnt_ppe
            std[variable] = fsnt_std - flnt_std

        else:
            ppe[variable], std[variable] = load_cesm_da(cesm_variable, component, subdir_name, proc_descr, cam_grid)
        
        if component == "clm2":
            ppe[variable] = ppe[variable].reindex(lat=cam_grid.lat, lon=cam_grid.lon, method="nearest", tolerance=0.05)
            std[variable] = std[variable].reindex(lat=cam_grid.lat, lon=cam_grid.lon, method="nearest", tolerance=0.05)

        ppe[variable] = ppe[variable].rename(variable)
        std[variable] = std[variable].rename(variable)

    return ppe, std


def get_hadcm(varlist, scenario, proc_type):
    """
    Parameters:
    -----------
    varlist : list
        List of variables
    scenario : string
        The HadCM3 PPE scenario ["a1b", "control]
    proc_type : string
        Type of processed output ["annual", "month_mean", "time_mean"]

    Returns:
    --------
    ppe : dict
        Dictionary with structure [variable] of processed HadCM3 PPE output
    std : dict
        Dictionary with structure [variable] of processed HadCM3 standard simulation output
    """
    # Source directory
    inpath = f"/glade/work/bbuchovecky/HadCM3_analysis/{scenario}"

    # Get the subdirectory name and file descriptor for the specified type of processed output
    subdir_name, proc_descr = get_processed_type_labels(proc_type)

    # Initialize data dictionaries
    ppe = {}
    std = {}

    print(f"HadCM3 {scenario} {proc_type}:")
    for variable in varlist:
        print(f"  {variable}")

        # Get the HadCM variable name
        hadcm_variable = HADCM_VAR_DICT[variable]

        # Check if the variable needs to be calculated

        if variable == "q_S":
            ppe[variable] = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_variable}.nc")[hadcm_variable]
            std[variable] = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_variable}.nc")[hadcm_variable]

            std[variable] = std[variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]
            ppe[variable] = ppe[variable] * UNIT_CONV["specific humidity [kg/kg -> g/kg]"]
        
        elif variable == "PmE":
            hadcm_prect = "precipitation_flux"
            prect_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_prect}.nc")[hadcm_prect]
            prect_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_prect}.nc")[hadcm_prect]

            hadcm_et = "surface_upward_water_flux"
            et_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_et}.nc")[hadcm_et]
            et_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_et}.nc")[hadcm_et]
        
            ppe[variable] = prect_ppe - et_ppe
            std[variable] = prect_std - et_std

        # Compute surface albedo from the downwelling surface shortwave flux and the net surface shortwave flux
        elif variable == "albedo_S":
            hadcm_fsds = "surface_downwelling_shortwave_flux_in_air"
            fsds_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsds}.nc")[hadcm_fsds]
            fsds_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsds}.nc")[hadcm_fsds]

            hadcm_fsns = "surface_net_downward_shortwave_flux"
            fsns_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsns}.nc")[hadcm_fsns]
            fsns_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsns}.nc")[hadcm_fsns]

            ppe[variable] = (fsds_ppe - fsns_ppe) / fsds_ppe
            std[variable] = (fsds_std - fsns_std) / fsds_std
        
        # Compute the longwave cloud forcing from the upwelling TOA longwave fullsky and clearsky fluxes
        elif variable == "LWCF":
            hadcm_flut = "toa_outgoing_longwave_flux"
            flut_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_flut}.nc")[hadcm_flut]
            flut_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_flut}.nc")[hadcm_flut]

            hadcm_flutc = "toa_outgoing_longwave_flux_assuming_clear_sky"
            flutc_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_flutc}.nc")[hadcm_flutc]
            flutc_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_flutc}.nc")[hadcm_flutc]

            ppe[variable] = flutc_ppe - flut_ppe
            std[variable] = flutc_std - flut_std

        # Compute the shortwave cloud forcing from the upwelling TOA shortwave fullsky and clearsky fluxes
        elif variable == "SWCF":
            hadcm_fsut = "toa_outgoing_shortwave_flux"
            fsut_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsut}.nc")[hadcm_fsut]
            fsut_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsut}.nc")[hadcm_fsut]

            hadcm_fsutc = "toa_outgoing_shortwave_flux_assuming_clear_sky"
            fsutc_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsutc}.nc")[hadcm_fsutc]
            fsutc_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsutc}.nc")[hadcm_fsutc]

            ppe[variable] = fsutc_ppe - fsut_ppe
            std[variable] = fsutc_std - fsut_std

        # Compute the net TOA radiative flux from the upwelling TOA shortwave and longwave fluxes, and the downwelling TOA shortwave flux
        elif variable == "Rnet_TOA":
            hadcm_fsdt = "toa_incoming_shortwave_flux"
            fsdt_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsdt}.nc")[hadcm_fsdt]
            fsdt_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsdt}.nc")[hadcm_fsdt]

            hadcm_flut = "toa_outgoing_longwave_flux"
            flut_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_flut}.nc")[hadcm_flut]
            flut_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_flut}.nc")[hadcm_flut]

            hadcm_fsut = "toa_outgoing_shortwave_flux"
            fsut_ppe = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_fsut}.nc")[hadcm_fsut]
            fsut_std = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_fsut}.nc")[hadcm_fsut]

            ppe[variable] = fsdt_ppe - flut_ppe - fsut_ppe
            std[variable] = fsdt_std - flut_std - fsut_std

        else:
            ppe[variable] = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_PPE.{scenario}.{proc_descr}.{hadcm_variable}.nc")[hadcm_variable]
            std[variable] = xr.open_dataset(f"{inpath}/{subdir_name}/HadCM3_STD.{scenario}.{proc_descr}.{hadcm_variable}.nc")[hadcm_variable]

        ppe[variable] = ppe[variable].rename(variable)
        std[variable] = std[variable].rename(variable)
    
    return ppe, std
