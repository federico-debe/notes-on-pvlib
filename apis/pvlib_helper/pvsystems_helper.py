import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from pvlib import pvsystem as pv

from common.enums import RackingType, TrackingType


class PVSystemHelper:

    def get_cec_inverters_by_kwp(
        self,
        kwp: float,
        dcac_min: float = 1.05,
        dcac_max: float = 1.35,
        r_target: float = 1.15,
    ):
        inv = pv.retrieve_sam('cecinverter').T  # rows = inverters
        pdc = kwp * 1000.0

        # max_n policy (same spirit as yours)
        if kwp <= 15:
            max_n = 1
        elif kwp <= 100:
            max_n = 2
        elif kwp <= 500:
            max_n = max(2, int(kwp / 50))
        else:
            max_n = max(4, int(kwp / 100))

        n_vals = np.arange(1, max_n + 1, dtype=float)
        lower = pdc / (n_vals * dcac_max)    # per-inverter Paco lower bound
        upper = pdc / (n_vals * dcac_min)    # per-inverter Paco upper bound

        paco = inv['Paco'].astype(float).values
        fits_any_n = np.any((paco >= lower[:, None]) & (paco <= upper[:, None]), axis=0)

        inv_fit = inv[fits_any_n].copy()

        # choose a "best" n near target DC/AC and compute the resulting DC/AC for ranking
        paco_fit = inv_fit['Paco'].astype(float).values
        n_best = np.clip(np.rint(pdc / (r_target * paco_fit)).astype(int), 1, max_n)
        dcac_best = pdc / (n_best * paco_fit)

        inv_fit['n_best'] = n_best
        inv_fit['dcac_best'] = dcac_best
        inv_fit['score'] = 1.0 - np.abs(dcac_best - r_target)  # higher is better

        # sort best-first and return in the original orientation (rows=parameters, cols=inverters)
        inv_fit = inv_fit.sort_values('score', ascending=False)
        return inv_fit.T

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
    def filter_cec_techs_by_voltage(
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
        kept_inverters = inverters.T.loc[inv.index[inv_keep_mask]].sort_values('score', ascending=False).T

        return kept_modules, kept_inverters

    @staticmethod
    def _estimate_cell_T(Tamb_hot=45.0, POA_hot=1000.0, wind_hot=1.0, Uc=20.0, Uv=6.0):
        # simple PVSyst-like linear model for hot case (good for screening)
        return Tamb_hot + POA_hot / (Uc + Uv*max(wind_hot, 0.1))

    def pick_stringing(
        self,
        module,
        inverter,
        *,
        t_cell_cold=-5.0,
        t_cell_hot=65.0,
        poa_w_per_m2=1000.0,
        mppt_low_key='Mppt_low',
        mppt_high_key='Mppt_high',
        vdcmax_key='Vdcmax',
        idcmax_key='Idcmax',
        dc_ac_ratio=1.20,
        voc_margin=0.97,
        prefer_mid_window=True,
        # optional extras; used only if all are provided
        n_mppt=None,
        idc_mppt_max_a=None,
        isc_mppt_max_a=None,
        inputs_per_mppt=None,
    ):

        # inverter essentials (present in CEC)
        mppt_low  = float(inverter[mppt_low_key])
        mppt_high = float(inverter[mppt_high_key])
        vdcmax    = float(inverter[vdcmax_key])
        idcmax    = float(inverter.get(idcmax_key, np.inf))
        paco      = float(inverter['Paco'])

        # module IV at extremes (use your helper)
        Vmp_cold, Imp_cold, Pmp_cold, Voc_cold, Isc_cold = PVSystemHelper.module_iv_at_complete(module, t_cell_cold, poa_w_per_m2)
        Vmp_hot,  Imp_hot,  Pmp_hot,  Voc_hot,  Isc_hot  = PVSystemHelper.module_iv_at_complete(module, t_cell_hot,  poa_w_per_m2)

        # --- series constraints (MPPT @ hot, Vdcmax @ cold) ---
        n_series_max_by_mppt = int(math.floor(mppt_high / max(Vmp_cold, 1e-6)))
        n_series_min_by_mppt = int(math.ceil (mppt_low  / max(Vmp_hot,  1e-6)))
        n_series_max_by_voc  = int(math.floor(voc_margin * vdcmax / max(Voc_cold, 1e-6)))

        n_series_min = max(1, n_series_min_by_mppt)
        n_series_max = min(n_series_max_by_mppt, n_series_max_by_voc)
        if n_series_min > n_series_max:
            raise ValueError(
                f"no valid series count: min={n_series_min} > max={n_series_max} "
                f"(mppt window / vdcmax constraints)"
            )

        # pick modules per string
        if prefer_mid_window:
            Vmp_nom, *_ = PVSystemHelper.module_iv_at_complete(module, 25.0, poa_w_per_m2)
            target = 0.5 * (mppt_low + mppt_high)
            candidates = [n for n in range(n_series_min, n_series_max + 1)
                        if (n * Vmp_cold <= mppt_high) and (n * Vmp_hot >= mppt_low)]
            n_series = min(candidates or [n_series_min], key=lambda n: abs(n * Vmp_nom - target))
        else:
            n_series = n_series_min

        # STC string power and strings from dc/ac target
        pmp_module_stc = float(module['STC'])  # you said STC is always present
        pmp_string_stc = n_series * pmp_module_stc
        dc_target_w = dc_ac_ratio * paco
        n_strings = max(1, int(round(dc_target_w / max(pmp_string_stc, 1e-9))))

        # total inverter DC current cap (Idcmax)
        total_imp_hot = n_strings * Imp_hot
        if total_imp_hot > idcmax:
            n_strings = max(1, int(math.floor(idcmax / max(Imp_hot, 1e-9))))
            total_imp_hot = n_strings * Imp_hot

        # --- optional: per-MPPT caps (apply only if all provided) ---
        if (n_mppt is not None) and (idc_mppt_max_a is not None) and (inputs_per_mppt is not None):
            strings_per_mppt = int(math.ceil(n_strings / max(1, int(n_mppt))))
            cap_by_imp  = int(math.floor(idc_mppt_max_a / max(Imp_hot, 1e-9)))
            per_mppt_cap = min(cap_by_imp, int(inputs_per_mppt))

            if (isc_mppt_max_a is not None):
                cap_by_isc = int(math.floor(isc_mppt_max_a / max(Isc_hot, 1e-9)))
                per_mppt_cap = min(per_mppt_cap, cap_by_isc)

            if strings_per_mppt > per_mppt_cap:
                n_strings = max(1, int(per_mppt_cap) * int(n_mppt))
                strings_per_mppt = int(math.ceil(n_strings / int(n_mppt)))
                total_imp_hot = n_strings * Imp_hot
        else:
            strings_per_mppt = None
            per_mppt_cap     = None

        details = dict(
            mppt_low=mppt_low, mppt_high=mppt_high, vdcmax=vdcmax, idcmax=idcmax, paco=paco,
            Vmp_cold=Vmp_cold, Vmp_hot=Vmp_hot, Voc_cold=Voc_cold,
            n_series_min_by_mppt=n_series_min_by_mppt,
            n_series_max_by_mppt=n_series_max_by_mppt,
            n_series_max_by_voc=n_series_max_by_voc,
            chosen_series=n_series,
            Pmp_module_stc=pmp_module_stc,
            Pmp_string_stc=pmp_string_stc,
            dc_target_W=dc_target_w,
            chosen_strings=n_strings,
            total_imp_hot=total_imp_hot,
            n_mppt=n_mppt,
            strings_per_mppt=strings_per_mppt,
            per_mppt_cap=per_mppt_cap,
        )
        return n_series, n_strings, details


    @staticmethod
    def module_iv_at_complete(module_params, temp_cell_C=25.0, poa=1000.0):
        """Return (Vmp, Imp, Pmp, Voc, Isc) at given cell temp using the CEC model.
        Robust to a few missing optional CEC fields."""
        # Required CEC keys (except we’ll default EgRef/dEgdT/Adjust)
        required = ['alpha_sc','a_ref','I_L_ref','I_o_ref','R_sh_ref','R_s']
        miss = PVSystemHelper._need(module_params, required)
        if miss:
            raise ValueError(f"CEC module parameters missing required keys: {miss}")

        # EgRef  = float(module_params.get('EgRef', 1.121))      # eV, crystalline Si typical
        # dEgdT  = float(module_params.get('dEgdT', -0.0002677)) # eV/K, crystalline Si typical
        # Adjust = float(module_params.get('Adjust', 1.0))

        calc = pv.calcparams_desoto(
            effective_irradiance=float(poa),
            temp_cell=float(temp_cell_C),
            alpha_sc=float(module_params['alpha_sc']),
            a_ref=float(module_params['a_ref']),
            I_L_ref=float(module_params['I_L_ref']),
            I_o_ref=float(module_params['I_o_ref']),
            R_sh_ref=float(module_params['R_sh_ref']),
            R_s=float(module_params['R_s']),
            # Adjust=Adjust, EgRef=EgRef, dEgdT=dEgdT
        )
        out = pv.singlediode(*calc, method='lambertw')
        return float(out['v_mp']), float(out['i_mp']), float(out['p_mp']), float(out['v_oc']), float(out['i_sc'])

    @staticmethod
    def _need(mp, keys):
        missing = [k for k in keys if k not in mp or mp[k] is None or (hasattr(mp[k], 'isna') and mp[k].isna())]
        return missing
