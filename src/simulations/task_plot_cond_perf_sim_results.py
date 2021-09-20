import pytask

from src.config import BLD
from src.config import SRC

import pandas as pd
import matplotlib.pyplot as plt
import json

@pytask.mark.parametrize("depends_on, produces, id",
    [
        (
            {
                "df_oracle_lengths": SRC / "simulations" / "helper_tables" / f"oracle_lengths_table_process_{type}.csv",
                "df_coverage": BLD / "simulations" / "cond_perf_simulations" / f"kernel_coverage_results_{type}.csv",
                "df_length": BLD / "simulations" / "cond_perf_simulations" / f"kernel_length_results_{type}.csv",
                "specs_dgp": SRC / "simulations" / "specs" / f"cond_sim_type_{type}.json",
            },
            {
                "build": BLD / "simulations" / "cond_perf_simulations" / f"plot_conditional_surface_{type}.pdf",
            },
            {
                "type": int(type),
            }
        )
        for type in [1,2,3,4]
    ],
)
def task_cond_perf_sim_results(depends_on, produces, id):
    if id["type"] == 1:
        from src.simulations.plotting_code_cond_sims.plot_code_sim_1 import plot_process
    elif id["type"] == 2:
        from src.simulations.plotting_code_cond_sims.plot_code_sim_2 import plot_process
    elif id["type"] == 3:
        from src.simulations.plotting_code_cond_sims.plot_code_sim_3 import plot_process
    elif id["type"] == 4:
        from src.simulations.plotting_code_cond_sims.plot_code_sim_4 import plot_process
    else: 
        raise ValueError("No correct type specified.")

    df_length= pd.read_csv(depends_on["df_length"])
    df_coverage = pd.read_csv(depends_on["df_coverage"])
    df_oracle_lengths = pd.read_csv(depends_on["df_oracle_lengths"])

    specs = json.loads(depends_on["specs_dgp"].read_text(encoding="utf-8"))
    plot_process(produces=produces["build"], df_length=df_length, 
                 df_coverage=df_coverage, df_oracle_lengths=df_oracle_lengths, 
                 specs_dgp=specs)
