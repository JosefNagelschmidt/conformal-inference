import pytask

from src.config import BLD
import pandas as pd
import matplotlib.pyplot as plt

def plot_power_res(binary_res, gauss_res, produces):
    
    fig, axs = plt.subplots(3,2,figsize=(8.27,8.7675), sharex=False, sharey=False)
    df_binary_grouped = binary_res.drop(["opt_estimator"], axis=1).groupby(by=["treatment_to_noise_ratio","main_effect_case"]).mean()
    df_gaussian_grouped = gauss_res.drop(["opt_estimator"], axis=1).groupby(by=["treatment_to_noise_ratio","main_effect_case"]).mean()

    plt.subplots_adjust(hspace=0.5)

    # first row left

    const_df_bin = df_binary_grouped.iloc[df_binary_grouped.index.get_level_values('main_effect_case') == "const"]
    const_df_bin.reset_index(level=0, inplace=True)
    const_df_bin.reset_index(level=0, inplace=True)
    const_df_bin = const_df_bin[const_df_bin["treatment_to_noise_ratio"] <= 7.00]

    axs[0, 0].plot(const_df_bin["treatment_to_noise_ratio"], const_df_bin["share_signif_oracles_given_ite_nonzero"])
    axs[0, 0].plot(const_df_bin["treatment_to_noise_ratio"], const_df_bin["share_signif_intervals_given_ite_nonzero"])
    axs[0, 0].plot(const_df_bin["treatment_to_noise_ratio"], const_df_bin["share_signif_asy_intervals_given_ite_nonzero"])

    axs[0, 0].grid(True)
    axs[0, 0].minorticks_on()

    axs[0, 0].set_ylabel("$m_1$ constant")
    axs[0, 0].set_xlabel("s")
    axs[0, 0].title.set_text("$\\tau_1(X)$")

    # first row right
    const_df_gaussian = df_gaussian_grouped.iloc[df_gaussian_grouped.index.get_level_values('main_effect_case') == "const"]
    const_df_gaussian.reset_index(level=0, inplace=True)
    const_df_gaussian.reset_index(level=0, inplace=True)
    const_df_gaussian = const_df_gaussian[const_df_gaussian["treatment_to_noise_ratio"] <= 15.00]


    axs[0, 1].plot(const_df_gaussian["treatment_to_noise_ratio"], const_df_gaussian["share_signif_oracles_given_ite_nonzero"])
    axs[0, 1].plot(const_df_gaussian["treatment_to_noise_ratio"], const_df_gaussian["share_signif_intervals_given_ite_nonzero"])
    axs[0, 1].plot(const_df_gaussian["treatment_to_noise_ratio"], const_df_gaussian["share_signif_asy_intervals_given_ite_nonzero"])

    axs[0, 1].grid(True)
    axs[0, 1].minorticks_on()
    axs[0, 1].set_xlabel("s")
    axs[0, 1].title.set_text('$\\tau_2(X)$')


    # second row left
    lin_df_bin = df_binary_grouped.iloc[df_binary_grouped.index.get_level_values('main_effect_case') == "linear"]
    lin_df_bin.reset_index(level=0, inplace=True)
    lin_df_bin.reset_index(level=0, inplace=True)
    lin_df_bin = lin_df_bin[lin_df_bin["treatment_to_noise_ratio"]  <= 7.00]

    axs[1, 0].set_ylabel("$m_2$ linear")

    axs[1, 0].plot(lin_df_bin["treatment_to_noise_ratio"] , lin_df_bin["share_signif_oracles_given_ite_nonzero"])
    axs[1, 0].plot(lin_df_bin["treatment_to_noise_ratio"] , lin_df_bin["share_signif_intervals_given_ite_nonzero"])
    axs[1, 0].plot(lin_df_bin["treatment_to_noise_ratio"], lin_df_bin["share_signif_asy_intervals_given_ite_nonzero"])

    axs[1, 0].set_xlabel("s")
    axs[1, 0].grid(True)
    axs[1, 0].minorticks_on()


    # second row right
    lin_df_gaussian = df_gaussian_grouped.iloc[df_gaussian_grouped.index.get_level_values('main_effect_case') == "linear"]
    lin_df_gaussian.reset_index(level=0, inplace=True)
    lin_df_gaussian.reset_index(level=0, inplace=True)
    lin_df_gaussian = lin_df_gaussian[lin_df_gaussian["treatment_to_noise_ratio"] <= 15.00]

    axs[1, 1].plot(lin_df_gaussian["treatment_to_noise_ratio"] , lin_df_gaussian["share_signif_oracles_given_ite_nonzero"])
    axs[1, 1].plot(lin_df_gaussian["treatment_to_noise_ratio"] , lin_df_gaussian["share_signif_intervals_given_ite_nonzero"])
    axs[1, 1].plot(lin_df_gaussian["treatment_to_noise_ratio"], lin_df_gaussian["share_signif_asy_intervals_given_ite_nonzero"])

    axs[1, 1].grid(True)
    axs[1, 1].minorticks_on()
    axs[1, 1].set_xlabel("s")


    # third row left
    non_lin_df_bin = df_binary_grouped.iloc[df_binary_grouped.index.get_level_values('main_effect_case') == "non-linear"]
    non_lin_df_bin.reset_index(level=0, inplace=True)
    non_lin_df_bin.reset_index(level=0, inplace=True)
    non_lin_df_bin = non_lin_df_bin[non_lin_df_bin["treatment_to_noise_ratio"]  <= 7.00]

    axs[2, 0].plot(non_lin_df_bin["treatment_to_noise_ratio"] , non_lin_df_bin["share_signif_oracles_given_ite_nonzero"])
    axs[2, 0].plot(non_lin_df_bin["treatment_to_noise_ratio"] , non_lin_df_bin["share_signif_intervals_given_ite_nonzero"])
    axs[2, 0].plot(non_lin_df_bin["treatment_to_noise_ratio"], non_lin_df_bin["share_signif_asy_intervals_given_ite_nonzero"])

    axs[2, 0].grid(True)
    axs[2, 0].minorticks_on()
    axs[2, 0].set_ylabel("$m_3$ non-linear")
    axs[2, 0].set_xlabel("s")


    # third row right

    const_asym_g_nonlin = df_gaussian_grouped.iloc[(df_gaussian_grouped.index.get_level_values('main_effect_case') == "non-linear")]
    const_asym_g_nonlin.reset_index(level=0, inplace=True)
    const_asym_g_nonlin.reset_index(level=0, inplace=True)
    const_asym_g_nonlin = const_asym_g_nonlin[const_asym_g_nonlin["treatment_to_noise_ratio"] <= 15.00]

    axs[2, 1].plot(const_asym_g_nonlin["treatment_to_noise_ratio"] , const_asym_g_nonlin["share_signif_oracles_given_ite_nonzero"])
    axs[2, 1].plot(const_asym_g_nonlin["treatment_to_noise_ratio"] , const_asym_g_nonlin["share_signif_intervals_given_ite_nonzero"])
    axs[2, 1].plot(const_asym_g_nonlin["treatment_to_noise_ratio"], const_asym_g_nonlin["share_signif_asy_intervals_given_ite_nonzero"])

    axs[2, 1].grid(True)
    axs[2, 1].minorticks_on()
    axs[2, 1].set_xlabel("s")

    plt.savefig(produces)


@pytask.mark.depends_on(
    {
        "df_binary": BLD / "simulations" / "power_simulations" / "df_results_binary.csv",
        "df_gaussian": BLD / "simulations" / "power_simulations" / "df_results_gaussian.csv",
    })
@pytask.mark.produces(BLD / "simulations" / "power_simulations" / "power_plot.pdf")
def task_power_simulations(depends_on, produces):
    binary_res = pd.read_csv(depends_on["df_binary"], index_col=[0,1,2,3,4,5])
    gauss_res = pd.read_csv(depends_on["df_gaussian"], index_col=[0,1,2,3,4,5])
    plot_power_res(binary_res, gauss_res, produces)