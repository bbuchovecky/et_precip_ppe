import numpy as np
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import et_precip_ppe.src.et_precip_ppe.analyze as ppealyz


m_dict = {
    "cesm": "o",
    "hadhst": "^",
    "hada1b": "s",
    "cmip": "D",
    "0.5xCO2": "^",
    "2xCO2": "s",
    "4xCO2": "d"
}
c_dict = {
    "cesm": "forestgreen",
    "hadhst": "orange",
    "hada1b": "steelblue",
    "cmip": "#000000",
    "0.5xCO2": "#000000",
    "2xCO2": "#000000",
    "4xCO2": "#000000",
    "cmip_mean": "deeppink",
    "lnd": "forestgreen",
    "ocn": "steelblue"
}
a_dict = {
    "cesm": 1,
    "hadhst": 1,
    "hada1b": 1,
    "cmip": 1
}
s_dict = {
    "cesm": 10,
    "hadhst": 20,
    "hada1b": 10,
    "cmip": 10,
    "0.5xCO2": 20,
    "2xCO2": 20,
    "4xCO2": 20
}
lw_dict = {
    "cesm": 2,
    "hadhst": 2,
    "hada1b": 2,
    "cmip": 2,
    "0.5xCO2": 2,
    "2xCO2": 2,
    "4xCO2": 2
}
label_dict = {
    "cesm": "CESM2 PI",
    "hadhst": "HadCM3C Hist",
    "hada1b": "HadCM3C A1B",
    "cmip": "CMIP6 CO$_2$",
    "0.5xCO2": "0.5$\\times$CO$_2$",
    "2xCO2": "2$\\times$CO$_2$",
    "4xCO2": "4$\\times$CO$_2$",
    "lnd": "Land",
    "ocn": "Ocean"
}
figenum = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i"
]
fw = "bold"
ax_lw = 0.8
axlabelsize = 8
legendsize = 8
axtitlesize = 8
export_dpi = 400


def setup_plotting_workspace():
    """Default matplotlib settings"""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 8,
        "font.weight": "normal",
        "axes.labelsize": axlabelsize,
        "axes.labelweight": "normal",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.fontsize": legendsize,
        "legend.frameon": False,
        "hatch.linewidth": 0.2,
    })


def plot_range_percent_change(exp, ctl, ax, cmap, vmin=0, vmax=100, lev_step=10, mask=None):
    """Creates a contour plot of the range in percent change across an ensemble of simulations"""
    dim = "member"
    pctchange = ( abs(exp - ctl) / ctl ) * 100
    pctchange = pctchange.max(dim=dim) - pctchange.min(dim=dim)

    if mask is not None:
        pctchange = pctchange.where(~np.isnan(mask))

    lon = pctchange.lon
    lat = pctchange.lat
    cyclic_pctchange, cyclic_lon = add_cyclic_point(pctchange, coord=lon)

    lev = np.arange(vmin, vmax+0.001, lev_step)

    cf = ax.contourf(
        cyclic_lon, lat, cyclic_pctchange, lev,
        cmap=cmap, vmin=vmin, vmax=vmax, extend="max",
        transform=ccrs.PlateCarree()
    )
    ax.coastlines(lw=0.5)

    return ax, cf


def plot_zonal_range_percent_change(exp, ctl, ax, mask=None, **kwargs):
    dim = "member"
    pctchange = ( abs(exp - ctl) / ctl ) * 100
    pctchange = pctchange.max(dim=dim) - pctchange.min(dim=dim)

    if mask is not None:
        pctchange = pctchange.where(~np.isnan(mask))
    
    pctchange = pctchange.mean(dim="lon")

    pctchange.plot(
        y="lat", ax=ax,
        lw=0.75,
        **kwargs
    )
    ax.set_ylim(-90,90)
    ax.set_title("")

    return ax


def plot_scatter_regression(axis, x, y, regress_func, key, alpha=0.05, do_scatter=True, return_stats=True, do_print=False, zorder=100):    
    assert isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray)
    assert len(x.dims) == 1 and x.dims == y.dims

    x = x.sortby(x)
    y = y.sortby(x)

    result = regress_func(x, y, alpha=alpha)

    if do_scatter:
        axis.scatter(x, y, s=s_dict[key], color=c_dict[key], marker=m_dict[key], lw=0, label=label_dict[key], zorder=zorder)
        axis.plot(x, result["slope"] * x + result["intercept"], color=c_dict[key], lw=lw_dict[key], zorder=zorder)
    else:
        axis.plot(x, result["slope"] * x + result["intercept"], color=c_dict[key], lw=lw_dict[key], label=label_dict[key], zorder=zorder)

    if return_stats:
        return result

    if do_print:
        print(key)
        print(f"y = {result["slope"]: 0.4f} * x + {result["intercept"]: 0.4f}")
        print(f"slope = {result["slope"]: 0.4f} ± {result["slope_ci_halfwidth"]:0.4f} at 95% CI")
        print(f"slope p-value = {result["slope_p_value"]: 0.4e}\n")


def plot_slope_map(axis, x, y, regress_func, alpha_fdr=0.05, sample_dim="member", mask_ocean=None, **kwargs):
    slope, intercept, slope_se, intercept_se, slope_ci, p_value = regress_func(x, y, sample_dim)
    p_value_fdr = ppealyz.calculate_pval_fdr(p_value, alpha_fdr)

    if mask_ocean is not None:
        assert isinstance(mask_ocean, (xr.Dataset, xr.DataArray))
        slope = slope.where(mask_ocean > 0)
        p_value = p_value.where(mask_ocean > 0)

    cf = axis.pcolormesh(slope.lon, slope.lat, slope.where(p_value < p_value_fdr), transform=ccrs.PlateCarree(), rasterized=True, **kwargs)

    axis.pcolor(
        slope.lon, slope.lat, slope.where(p_value >= p_value_fdr),
        cmap=ListedColormap(["white"]),
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    axis.pcolor(
        slope.lon, slope.lat, slope.where(p_value >= p_value_fdr),
        hatch="xxxxxxx",
        cmap=ListedColormap(["none"]),
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    axis.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    axis.set_global()
    axis.coastlines(lw=plt.rcParams["axes.linewidth"])

    return cf


def plot_corr_map(axis, x, y, alpha_fdr=0.05, sample_dim="member", mask_ocean=None, **kwargs):
    corr = xs.pearson_r(x, y, dim=sample_dim)
    p_value = xs.pearson_r_p_value(x, y, dim=sample_dim)
    p_value_fdr = ppealyz.calculate_pval_fdr(p_value, alpha_fdr)

    if mask_ocean is not None:
        assert isinstance(mask_ocean, (xr.Dataset, xr.DataArray))
        corr = corr.where(mask_ocean > 0)
        p_value = p_value.where(mask_ocean > 0)

    cf = axis.pcolormesh(corr.lon, corr.lat, corr.where(p_value < p_value_fdr), transform=ccrs.PlateCarree(), rasterized=True, **kwargs)

    axis.pcolor(
        corr.lon, corr.lat, corr.where(p_value >= p_value_fdr),
        cmap=ListedColormap(["white"]),
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    axis.pcolor(
        corr.lon, corr.lat, corr.where(p_value >= p_value_fdr),
        hatch="xxxxxxx",
        cmap=ListedColormap(["none"]),
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    axis.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    axis.set_global()
    axis.coastlines(lw=plt.rcParams["axes.linewidth"])

    return cf
