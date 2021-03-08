import numpy as np
from scipy import stats
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, Callable, Tuple, Any, Dict, TypeVar
from numpy.typing import ArrayLike

_bkg_color = "#212121"
_fgr_color = "#eeffff"
soft_dark_style = {
    "axes.axisbelow": "line",
    "axes.edgecolor": _fgr_color,
    "axes.facecolor": _bkg_color,
    "axes.labelcolor": _fgr_color,
    "boxplot.boxprops.color": _fgr_color,
    "boxplot.capprops.color": _fgr_color,
    "boxplot.flierprops.color": _fgr_color,
    "boxplot.flierprops.markeredgecolor": _fgr_color,
    "boxplot.whiskerprops.color": _fgr_color,
    "figure.edgecolor": _bkg_color,
    "figure.facecolor": _bkg_color,
    "grid.color": _fgr_color,
    "lines.color": _fgr_color,
    "patch.edgecolor": _fgr_color,
    "savefig.edgecolor": _bkg_color,
    "savefig.facecolor": _bkg_color,
    "text.color": _fgr_color,
    "xtick.color": _fgr_color,
    "ytick.color": _fgr_color,
}


def joint_hist(
    x_sample: ArrayLike,
    y_sample: ArrayLike,
    ax: Optional[Axes] = None,
    *,
    x_pdf: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    y_pdf: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    pdf_ax_size: Union[int, str] = "15%",
    bins: Union[int, ArrayLike] = 64,
    use_contourf: bool = True,
    hist_kwargs: Optional[Dict[str, Any]] = None,
    fill_between_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    ret_ax: bool = False
) -> Union[
    Tuple[ArrayLike, ArrayLike, ArrayLike],
    Tuple[ArrayLike, ArrayLike, ArrayLike, Axes, Axes],
]:
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    if x_pdf is None:
        x_pdf = stats.gaussian_kde(x_sample)
    if y_pdf is None:
        y_pdf = stats.gaussian_kde(y_sample)
    if hist_kwargs is None:
        hist_kwargs = {}
    if fill_between_kwargs is None:
        fill_between_kwargs = {}

    counts, xbins, ybins = np.histogram2d(x_sample, y_sample, bins, density=True)

    divider = make_axes_locatable(ax)
    ax_x_pdf = divider.append_axes("top", size=pdf_ax_size, pad=0, sharex=ax)
    ax_y_pdf = divider.append_axes("right", size=pdf_ax_size, pad=0, sharey=ax)

    if use_contourf:
        im = ax.contourf(
            counts.T, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], **hist_kwargs
        )
    else:
        im = ax.pcolormesh(xbins, ybins, counts.T, **hist_kwargs)

    x = np.linspace(*ax.get_xlim(), 512)
    btwn = ax_x_pdf.fill_between(
        x,
        0,
        x_pdf(x),
        **{
            "facecolor": ax.get_facecolor(),
            "edgecolor": ax.spines["top"].get_edgecolor(),
            "hatch": "x",
            "clip_on": False,
            **fill_between_kwargs,
        }
    )
    ax_x_pdf.set_ylim(0, None)
    ax_x_pdf.set_axis_off()
    ax_x_pdf.set_frame_on(False)

    y = np.linspace(*ax.get_ylim(), 512)
    ax_y_pdf.fill_betweenx(
        y,
        0,
        y_pdf(y),
        **{
            "facecolor": ax.get_facecolor(),
            "edgecolor": ax.spines["right"].get_edgecolor(),
            "hatch": "x",
            "clip_on": False,
            **fill_between_kwargs,
        }
    )
    ax_y_pdf.set_xlim(0, None)
    ax_y_pdf.set_axis_off()
    ax_y_pdf.set_frame_on(False)

    if title:
        ax_x_pdf.set_title(title)

    if ret_ax:
        return counts, xbins, ybins, ax_x_pdf, ax_y_pdf
    else:
        return counts, xbins, ybins
