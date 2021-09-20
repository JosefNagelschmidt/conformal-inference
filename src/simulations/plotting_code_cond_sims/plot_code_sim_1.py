import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.conformal_methods.utils import generate_X_fixed_positions, generate_y_fixed_positions, get_conditional_variances


def sample_linear_part(specs_dgp):
    X = generate_X_fixed_positions(n=150, p=specs_dgp["p"], X_dist=specs_dgp["X_dist"], cor=specs_dgp["X_correlation"], 
                                    standardize=bool(int(specs_dgp["standardized_X"])), uniform_upper=specs_dgp["uniform_upper"])

    y, eps, sigma_vec, mu, beta = generate_y_fixed_positions(X_mat=X, eps_dist=specs_dgp["eps_dist"], 
                                                                error_type=specs_dgp["error_type"], 
                                                                functional_form=specs_dgp["functional_form"], 
                                                                non_zero_beta_count=specs_dgp["non_zero_beta_count"])
    linear_part = X @ beta
    return linear_part


def plot_process(produces, df_length, df_coverage, df_oracle_lengths, specs_dgp):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(11.5,6), constrained_layout=False)

    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["mean_based_cond_coverage"], 'r', label='mean_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["w_mean_based_cond_coverage"], label='w_mean_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["quantile_based_cond_coverage"], label='quantile_based')
    ax2.plot(df_coverage["cond_variance_y_grid"], df_coverage["cdf_based_cond_coverage"], label='cdf_based')


    ax2.axhline(y=0.9, color='black', linestyle='-.')
    ax2.axhline(y=1.0, color='black', linestyle='-.')

    sample = sample_linear_part(specs_dgp)
    ax2.plot(sample, [0.31]*len(sample), '|', color='k')

    # set x-axis label
    ax2.set_xlabel("β'X")
    ax2.set_xlim([-5.5, 5.5])
    ax2.set_ylim([0.3, 1.1])
    # set y-axis label
    ax2.set_ylabel("conditional coverage",color="black")
    # twin object for two different y-axis on the sample plot
    ax2_2=ax2.twinx()
    x_grid, cond_var = get_conditional_variances(process_type=1)

    # make a plot with different y-axis using second axis object
    ax2_2.plot(x_grid, cond_var,':', color="black")

    ax2_2.set_ylabel("conditional variance (dotted)",color="black")
    ax2_2.set_xlim([-5.5, 5.5])
    ax2_2.set_ylim([-5, 100])
    ##################################

    ax1.plot(df_length["cond_variance_y_grid"], df_length["mean_based_cond_length"], 'r', label='mean based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["w_mean_based_cond_length"], label='weighted mean based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["quantile_based_cond_length"], label='quantile based')
    ax1.plot(df_length["cond_variance_y_grid"], df_length["cdf_based_cond_length"], label='cdf based')

    # set x-axis label
    ax1.set_xlabel("β'X")
    # set y-axis label
    ax1.set_ylabel("conditional length",color="black")
    #ax.set_xlim([0.0, 1])
    ax1.set_ylim([0, 30])
    ax1.set_xlim([-5.5, 5.5])
    
    df_oracle_length = np.array(df_oracle_lengths)
    x_ax = df_oracle_length[:,0]
    oracle_length = df_oracle_length[:,1]
    ax1.plot(x_ax, oracle_length, ':', color="black", label='oracle')
    ax1.legend()
 
    plt.savefig(produces)