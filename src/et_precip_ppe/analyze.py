import numpy as np
import xarray as xr
import scipy.stats as stats
import scipy.odr as odr


def calculate_pval_fdr(
    ps : xr.DataArray | np.ndarray,
    alpha_fdr : float) -> float:
    """
    Calculates the adjusted p-value threshold given a
    specified false discovery rate (FDR) control level.

    Wilks, D. S. (2016). “The Stippling Shows Statistically Significant Grid Points”:
    How Research Results are Routinely Overstated and Overinterpreted, and What to Do about It.
    Bulletin of the American Meteorological Society, 97(12), 2263–2273.
    https://doi.org/10.1175/BAMS-D-15-00267.1

    Parameters:
    -----------
    ps : xarray.DataArray | numpy.ndarray
        Array of p-values from local hypothesis tests at each grid point
    alpha_fdr : float
        The specified control level for the false discovery rate

    Returns:
    --------
    pval_fdr : float
        The adjusted p-value threshold
    """

    # Convert DataArray to NumPy array if necessary
    if isinstance(ps, xr.DataArray):
        ps = ps.values

    # Sort the p-values
    pval_sorted = np.sort(ps.flatten())
    n = pval_sorted.shape[0]

    # Select p-values below the FDR control level
    pval_sorted_subset = np.where(pval_sorted <= (np.linspace(1, n, n) / n) * alpha_fdr)[0]

    # Select the maximum p-value below the FDR control level as the FDR adjusted p-value
    if pval_sorted_subset.size > 0:
        pval_fdr = pval_sorted[pval_sorted_subset].max()
    else:
        print('no p-values above the FDR control level')
        pval_fdr = np.nan
    
    return pval_fdr


def _ols_single(x, y, alpha=0.05):
    """ Core function for computing ordinary least squares (OLS) regression """
    # Coerce to 1D arrays and drop NaNs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

     # Exit if not enough data
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Perform the regression output
    ols = stats.linregress(x, y)
    slope = ols.slope
    intercept = ols.intercept
    slope_se = ols.stderr
    intercept_se = ols.intercept_stderr

    # Estimate the degrees of freedom
    p_free = 2
    dof = max(len(x) - p_free, 1)

    # Compute the critical t-value and the confidence interval for the slope
    tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
    slope_ci_halfwidth = tcrit * slope_se

    # Wald test for H0: slope == 0
    t_stat = slope / slope_se
    slope_p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof))

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def ols_single(x, y, alpha=0.05):
    """ Wrapper of OLS for 1D arrays """
    out = _ols_single(x, y, alpha=alpha)
    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    result = dict(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        intercept_se=intercept_se,
        slope_ci_halfwidth=slope_ci_halfwidth,
        slope_p_value=slope_p_value,
    )
    return result


def ols_field(x_da, y_da, sample_dim, alpha=0.05):
    """ Wrapper of OLS for multidimensional xarray datasets """
    args = [x_da, y_da]
    core_dims = [[sample_dim], [sample_dim]]

    out = xr.apply_ufunc(
        _ols_single, *args,
        input_core_dims=core_dims,
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        join="inner",
        output_dtypes=[float, float, float, float, float, float],
        kwargs=dict(alpha=alpha)
    )

    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    slope.name              = "slope",
    intercept.name          = "intercept"
    slope_se.name           = "slope_se"
    intercept_se.name       = "intercept_se"
    slope_ci_halfwidth.name = "slope_ci_halfwidth"
    slope_p_value.name      = "slope_p_value"

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def _odr_single(x, y, alpha=0.05):
    """ Core function for computing orthogonal distance regression (ODR) regression """
    # Coerce to 1D arrays and drop NaNs
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

     # Exit if not enough data
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Linear function y = B_0 * x + B_1
    def f(B, X):
        return B[0] * X + B[1]

    # Use OLS to get initial parameter values
    ols = stats.linregress(x, y)
    beta0 = np.array([ols.slope, ols.intercept], dtype=float)

    # Create the ODR model and perform the regression
    model = odr.Model(f)
    data = odr.Data(x, y)
    myodr = odr.ODR(data, model, beta0=beta0)
    out = myodr.run()

    # Store the regression output
    slope, intercept = out.beta[0], out.beta[1]
    slope_se, intercept_se = out.sd_beta[0], out.sd_beta[1]

    # Estimate the degrees of freedom
    p_free = 2
    dof = max(len(x) - p_free, 1)

    # Compute the critical t-value and the confidence interval for the slope
    tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
    slope_ci_halfwidth = tcrit * slope_se

    # Wald test for H0: slope == 0
    t_stat = slope / slope_se
    slope_p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof))

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def odr_single(x, y, alpha=0.05):
    """ Wrapper of ODR for 1D arrays """
    out = _odr_single(x, y, alpha=alpha)
    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    result = dict(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        intercept_se=intercept_se,
        slope_ci_halfwidth=slope_ci_halfwidth,
        slope_p_value=slope_p_value,
    )
    return result


def odr_field(x_da, y_da, sample_dim, alpha=0.05):
    """ Wrapper of ODR for multidimensional xarray datasets """
    args = [x_da, y_da]
    core_dims = [[sample_dim], [sample_dim]]

    out = xr.apply_ufunc(
        _odr_single, *args,
        input_core_dims=core_dims,
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        join="inner",
        output_dtypes=[float, float, float, float, float, float],
        kwargs=dict(alpha=alpha),
    )

    slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value = out
    slope.name              = "slope",
    intercept.name          = "intercept"
    slope_se.name           = "slope_se"
    intercept_se.name       = "intercept_se"
    slope_ci_halfwidth.name = "slope_ci_halfwidth"
    slope_p_value.name      = "slope_p_value"

    return slope, intercept, slope_se, intercept_se, slope_ci_halfwidth, slope_p_value


def normalize_to_centered_range(arr, scale=1):
    """
    Takes an array and divides by the infinity norm which 
    normalizes the array to the range [-scale, scale].
    
    The sign of each value is preserved and if the input
    array is an array of arrays, the relative magnitude between
    the values in each array is also preserved. This allows
    comparison between the normalized arrays.

    Parameters:
    -----------
    arr : list | np.ndarray | xr.DataArray
        Array of values, or an array of arrays.
    scale : float
        Value that sets the output normalization range.

    Returns:
    --------
    norm_arr : np.ndarray
        Array of normalized values, same shape as the
        input array.
    """
    if isinstance(arr, list):
        arr = np.array(arr)

    if isinstance(arr, xr.DataArray):
        arr = arr.values

    assert isinstance(arr, np.ndarray)

    shape_arr = arr.shape
    flat_arr = arr.flatten()
    norm_arr = flat_arr/abs(flat_arr).max()*scale
    norm_arr = norm_arr.reshape(shape_arr)

    return norm_arr


def mass_fraction_co2_to_ppmv(mass_fraction_co2):
    co2_molar_mass    = 44         # [g mol-1]
    atmos_total_mass  = 5.1361e21  # [g]
    atmos_total_moles = 1.77678e20 # [mol]

    co2_total_moles = (mass_fraction_co2 * atmos_total_mass) / co2_molar_mass
    co2_ppmv = co2_total_moles / atmos_total_moles * 1e6

    return co2_ppmv


def mass_fraction_to_ppm(mass_fraction_species, molar_mass_species, molar_mass_air=28.948):
    """
    Convert the mass fraction of a gas species to the atmospheric mixing ratio.
    molar_mass_air = 0.7808 * M_N2 + 0.2095 * M_O2 + 0.0093 * M_Ar = 28.948 g/mol
    M_N2 = 28.013 g/mol
    M_O2 = 32.000 g/mol
    M_Ar = 39.950 g/mol
    """
    moles_species = mass_fraction_species / molar_mass_species
    moles_air = (1 - mass_fraction_species) / molar_mass_air
    mole_fraction =  moles_species / (moles_species + moles_air)
    return mole_fraction * 1e6


def weighted_average(da, weights):
    return (da * weights).sum(dim=['lat', 'lon']) / weights.sum(dim=['lat', 'lon'])


def co2_radiative_forcing(c, c_0):
    """ Table 8.SM.1 in Chapter 8: Anthropogenic and Natural Radiative Forcing, IPPC 2013 """
    return 5.35 * np.log(c / c_0)
