import matplotlib.pyplot as plt
import numpy as np
from src.conformal_methods.utils import get_conditional_variances


def plot_process(produces, df_length, df_coverage, df_oracle_lengths, specs_dgp):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11.5,6), constrained_layout=False)

    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["mean_based_cond_coverage"], 'r', label='mean_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["w_mean_based_cond_coverage"], label='w_mean_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["quantile_based_cond_coverage"], label='quantile_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["cdf_based_cond_coverage"], label='cdf_based')

    ax2.axhline(y=0.9, color='black', linestyle='-.')
    ax2.axhline(y=1.0, color='black', linestyle='-.')

    # set x-axis label
    ax2.set_xlabel("X")
    ax2.set_xlim([0, 1])
    # set y-axis label
    ax2.set_ylabel("conditional coverage",color="black")
    # twin object for two different y-axis on the sample plot
    ax2_2=ax2.twinx()
    x_grid, cond_var = get_conditional_variances(process_type=3)
    # make a plot with different y-axis using second axis object
    ax2_2.plot(x_grid, cond_var,':', color="black")
    ax2_2.set_ylabel("conditional variance (dotted)",color="black")
    ##################################
    ax1.plot(df_length["cond_variance_y_grid"], df_length["mean_based_cond_length"], 'r', label='mean based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["w_mean_based_cond_length"], label='weighted mean based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["quantile_based_cond_length"], label='quantile based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["cdf_based_cond_length"], label='cdf based')
    # set x-axis label
    ax1.set_xlabel("X")
    # set y-axis label
    ax1.set_ylabel("conditional length",color="black")
    ax1.set_xlim([0, 1])

    df_oracle_length = np.array(df_oracle_lengths)
    x_ax = df_oracle_length[:,0]
    oracle_length = df_oracle_length[:,1]
    ax1.plot(x_ax, oracle_length, ':', color="black", label='oracle')
    ax1.legend()

    plt.savefig(produces)