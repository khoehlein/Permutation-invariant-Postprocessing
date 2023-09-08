import argparse

import numpy as np

from experiments.baselines.pp_drn import drn_pp
import pandas as pd


DYNAMIC_PREDICTORS = [
    'VMAX_10M',  # 'VMAX_10M_LS', 'VMAX_10M_LS_S', 'VMAX_10M_MS', 'VMAX_10M_MS_S',
    'U_10M',  # 'U_10M_LS', 'U_10M_LS_S', 'U_10M_MS', 'U_10M_MS_S',
    'U500', 'U1000', 'U700', 'U850', 'U950',
    'V_10M',
    'V500', 'V1000', 'V700', 'V850', 'V950',
    'WIND_10M',
    'WIND500', 'WIND1000',
    'OMEGA500', 'OMEGA1000',  'OMEGA700', 'OMEGA850', 'OMEGA950',
    'T_G',  # 'T_G_LS', 'T_G_LS_S', 'T_G_MS', 'T_G_MS_S',
    'T_2M',  # 'T_2M_LS', 'T_2M_LS_S', 'T_2M_MS', 'T_2M_MS_S',
    'T500', 'T1000',# 'T700', 'T850', 'T950', 'T1000',
    'TD_2M', # 'TD_2M_LS', 'TD_2M_LS_S', 'TD_2M_MS', 'TD_2M_MS_S',
    'RELHUM500', 'RELHUM1000', 'RELHUM700', 'RELHUM850', 'RELHUM950',
    'TOT_PREC',
    'RAIN_GSP',
    'SNOW_GSP',
    'W_SNOW',
    'W_SO1', 'W_SO2', 'W_SO6', 'W_SO18', 'W_SO54',
    'CLCT',
    'CLCL',
    'CLCM',
    'CLCH',
    'HBAS_SC',
    'HTOP_SC',
    'ASOB_S',  # 'ASOB_S_LS', 'ASOB_S_LS_S', 'ASOB_S_MS', 'ASOB_S_MS_S',
    'ATHB_S',  # 'ATHB_S_LS', 'ATHB_S_LS_S', 'ATHB_S_MS', 'ATHB_S_MS_S',
    'ALB_RAD', # 'ALB_RAD_LS', 'ALB_RAD_LS_S', 'ALB_RAD_MS', 'ALB_RAD_MS_S',
    'PMSL', # 'PMSL_LS', 'PMSL_LS_S', 'PMSL_MS', 'PMSL_MS_S',
    'FI500', 'FI1000',  'FI700', 'FI850', 'FI950',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type=str, required=True)
    parser.add_argument('--data-test', type=str, required=True)
    parser.add_argument('--log-path', type=str, default=None)
    args = vars(parser.parse_args())

    df_train = pd.read_csv(args['data_train'], index_col=0)
    df_test = pd.read_csv(args['data_test'], index_col=0)

    pred_vars = [p + '_mean' for p in DYNAMIC_PREDICTORS] + ['alt', 'orog', 'yday', 'loc_bias', 'loc_cover', 'lat', 'lon']

    loc_id_vec = np.arange(175)

    pred = drn_pp(
        train=df_train,
        X=df_test,
        i_valid=None,
        loc_id_vec=loc_id_vec,
        pred_vars=pred_vars,
        nn_ls={
            'n_sim': 20,
            'nn_verbose': True
        },
        output_path=args['log_path']
    )

    scores_pp: pd.Series = pred['scores_pp']
    print(scores_pp.describe())


if __name__ == '__main__':
    main()
