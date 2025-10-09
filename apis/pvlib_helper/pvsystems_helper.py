import numpy as np
import pandas as pd

from pvlib import pvsystem as pv

from models.pv_custom.enums import RackingType, TrackingType


class PVSystemHelper:

    def get_cec_inverters_by_kwp(
            self, 
            kwp,
            dcac_max=1.35
        ):

        inv = pv.retrieve_sam('cecinverter').T.copy()
        Pdc = kwp * 1000.0

        if kwp <= 15:
            max_n = 1
        elif kwp <= 100:
            max_n = 2
        elif kwp <= 500:
            max_n = max(2, int(kwp / 50))
        else:
            max_n = max(4, int(kwp / 100))

        n_vals = np.arange(1, max_n + 1, dtype=float)
        lower = Pdc / (n_vals * dcac_max)
        upper = Pdc / (n_vals)

        Paco = inv['Paco'].astype(float).values  
        fits_any_n = np.any(
            (Paco >= lower[:, None]) & (Paco <= upper[:, None]),
            axis=0
        )

        has_windows = inv[['Vdcmax', 'Mppt_low', 'Mppt_high']].notna().all(axis=1)

        filtered = inv[fits_any_n & has_windows].T 
        return filtered

    def get_cec_modules_by_tech_choice(self, technology) -> pd.DataFrame:
        cecmodules = pv.retrieve_sam('cecmod')
        tech_modules = cecmodules.T[cecmodules.T['Technology'] == technology].T

        return tech_modules

    def filter_legacy_or_out_of_scope_cec_modules(self, filtered_modules):
        name = filtered_modules.index.astype(str).str.lower()
        drop_mask = (
            name.T.str.contains('cpv|concentrat|bipv|laminat|tile|transparent')
        )
        filtered_modules = filtered_modules[~drop_mask]
        filtered_modules = filtered_modules.T[filtered_modules.T['STC'] >= 240].T

        return filtered_modules

    def filter_cec_modules_by_racking_type(self, filtered_modules: pd.DataFrame, racking_type: RackingType) -> pd.DataFrame:

        if racking_type in (RackingType.open_rack, RackingType.freestanding):
            filtered_modules = filtered_modules.T[filtered_modules.T['STC'] >= 450].T
            filtered_modules = filtered_modules.T[filtered_modules.T['STC'] / (filtered_modules.T['A_c'] * 1000) >= 0.19].T
        elif racking_type in (RackingType.close_mount, RackingType.semi_integrated):
            filtered_modules = filtered_modules.T[filtered_modules.T['A_c'] <= 2.2].T
            filtered_modules = filtered_modules.T[filtered_modules.T['Bifacial'] == 0].T
            filtered_modules = filtered_modules.T[filtered_modules.T['STC'] / (filtered_modules.T['A_c'] * 1000) >= 0.18].T
        elif racking_type in (RackingType.insulated_back, RackingType.insulated):
            filtered_modules = filtered_modules.T[filtered_modules.T['A_c'] <= 2.0].T
            filtered_modules = filtered_modules.T[filtered_modules.T['Bifacial'] == 0].T
            filtered_modules = filtered_modules.T[filtered_modules.T['STC'] / (filtered_modules.T['A_c'] * 1000) >= 0.19].T

        return filtered_modules

    @staticmethod
    def filter_cec_modules_by_voltage(
        cecmodules: pd.DataFrame,
        inverters: pd.DataFrame,
        u_v,
        u_c,
        hot_temperature_ambient: float = 45.0,
        cold_temperature_ambient: float = -10.0,
        POA_hot: float = 1000.0,
        wind_hot: float = 1.0,
        eps: float = 1e-6,
    ):

        hot_temperature_cell = PVSystemHelper._estimate_cell_T(hot_temperature_ambient, POA_hot, wind_hot, u_c, u_v)
        Vmp_hot  = cecmodules['V_mp_ref'] + cecmodules['beta_oc'] * (hot_temperature_cell - 25.0)
        Voc_cold = cecmodules['V_oc_ref'] + cecmodules['beta_oc'] * (cold_temperature_ambient - 25.0)

        cecmodules['Vmp_hot']  = Vmp_hot
        cecmodules['Voc_cold'] = Voc_cold

        if cecmodules.empty or inverters.empty:
            return cecmodules.iloc[0:0], inverters.iloc[0:0]  
        inv = inverters.T[['Mppt_low','Mppt_high','Vdcmax']].astype(float).copy()
        inv = inv.dropna()
        if inv.empty:
            return cecmodules.iloc[0:0], inverters.iloc[0:0]

        Vmp_hot_arr  = cecmodules['Vmp_hot'].astype(float).to_numpy(dtype=np.float64).reshape(-1, 1)   # (M,1)
        Voc_cold_arr = cecmodules['Voc_cold'].astype(float).to_numpy(dtype=np.float64).reshape(-1, 1)  # (M,1)
        Vlo = inv['Mppt_low' ].to_numpy(dtype=np.float64).reshape(1, -1)  # (1,N)
        Vhi = inv['Mppt_high'].to_numpy(dtype=np.float64).reshape(1, -1)  # (1,N)
        Vdc= inv['Vdcmax'    ].to_numpy(dtype=np.float64).reshape(1, -1)  # (1,N)

        n_min       = np.ceil( Vlo / np.maximum(Vmp_hot_arr, eps) )
        n_max_mppt  = np.floor( Vhi / np.maximum(Vmp_hot_arr, eps) )
        n_max_vdc   = np.floor( (Vdc - eps) / np.maximum(Voc_cold_arr, eps) )
        n_max       = np.minimum(n_max_mppt, n_max_vdc)

        feasible = (n_min <= n_max) & (n_min >= 1)

        mod_keep_mask = feasible.any(axis=1)  # (M,)
        inv_keep_mask = feasible.any(axis=0)  # (N,)

        kept_modules   = cecmodules.loc[cecmodules.index[mod_keep_mask]].T
        kept_inverters = inverters.T.loc[inv.index[inv_keep_mask]].T

        return kept_modules, kept_inverters

    @staticmethod
    def _estimate_cell_T(Tamb_hot=45.0, POA_hot=1000.0, wind_hot=1.0, Uc=20.0, Uv=6.0):
        # simple PVSyst-like linear model for hot case (good for screening)
        return Tamb_hot + POA_hot / (Uc + Uv*max(wind_hot, 0.1))
