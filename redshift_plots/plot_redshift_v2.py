from photoz_utils import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def popout_plot_predictions(z_photo, z_spec, categories, values, save_path="photoz_vs_specz.png"):
    """
    Plot predicted vs. true (spectroscopic) redshifts as a 2D histogram
    with a 1-to-1 reference line, including a zoomed-in inset (0–0.4).
    """

    # Aesthetic settings
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 24,
        "axes.titlesize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix"
    })

    # === Main plot === #
    fig, ax = plt.subplots(figsize=(10, 8))

    h = ax.hist2d(
        z_spec, z_photo,
        bins=200,
        norm=LogNorm(),
        cmap="viridis"
    )
    cbar = plt.colorbar(h[3], ax=ax)
    #cbar.set_label("Counts")

    # 1:1 line
    lims = [0, max(np.max(z_spec), np.max(z_photo), 4)]
    ax.plot(lims, lims, color="red", lw=2, ls="--", label="1:1 line")

    # Labels
    ax.set_xlabel(r"$z_{\mathrm{spec}}$", fontsize=35)
    ax.set_ylabel(r"$z_{\mathrm{pred}}$", fontsize=35)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()

    # === Inset === #
    axins = inset_axes(ax, width="50%", height="50%", loc="center left",
                       bbox_to_anchor=(-0.75, -0.32, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.hist2d(
        z_spec, z_photo,
        bins=800,  # finer bins
        norm=LogNorm(),
        cmap="viridis"
    )
    axins.plot([0, 0.4], [0, 0.4], color="red", lw=2, ls="--")
    axins.set_xlim(0, 0.4)
    axins.set_ylim(0, 0.4)
    axins.set_xticks([0, 0.2, 0.4])
    axins.set_yticks([0, 0.2, 0.4])
    axins.tick_params(labelsize=18)

    # Connector lines showing the zoomed region
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5", lw=0.75)

    # === Bar plot in top-left corner === #
    # Add a new inset axis, positioned relative to the figure (not ax)
    bar_ax = fig.add_axes([-0.335, 0.57, 0.33, 0.33])
    # [left, bottom, width, height] in figure coordinates (0–1)

    bar_ax.bar(categories, values, color="rebeccapurple")
    bar_ax.set_title("Low-Redshift Prediction Scatter", fontsize=20)
    bar_ax.tick_params(axis="x", labelsize=18)
    bar_ax.tick_params(axis="y", labelsize=18)
    bar_ax.axhline(0, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    mmae_preds = pd.read_csv('mmae_redshift_predictions_v2.csv')
    z_photo = mmae_preds['y_pred']
    z_spec = mmae_preds['y_test']

    astroclip_preds = pd.read_csv('mmae_redshift_predictions_v1.csv')
    astroclip_photo = astroclip_preds['y_pred']
    astroclip_spec = astroclip_preds['y_test']

    mmae_metrics = get_point_metrics(z_photo, z_spec, binrange=np.linspace(np.min(astroclip_photo), np.max(astroclip_photo), 2))
    astroclip_metrics = get_point_metrics(astroclip_photo, astroclip_spec, binrange=np.linspace(np.min(astroclip_photo), np.max(astroclip_photo), 2))
    mmae_bias = mmae_metrics['bias_conv'][0]
    astroclip_bias = astroclip_metrics['bias_conv'][0]
    mmae_scatter = mmae_metrics['scatter_conv'][0]
    astroclip_scatter = astroclip_metrics['scatter_conv'][0]

    popout_plot_predictions(z_photo, z_spec, ["MMAE_V1", "MMAE_V2"], [astroclip_scatter, mmae_scatter], save_path="photoz_vs_specz_v3.png")


if __name__ == '__main__':
    main()