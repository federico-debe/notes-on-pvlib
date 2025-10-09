import numpy as np
import pandas as pd

from pvlib import pvsystem as pv

class PVSystemHelper:

    @staticmethod
    def filter_cec_modules(technology) -> pd.DataFrame:
        cecmodules = pv.retrieve_sam('cecmod')
        return cecmodules.T[cecmodules.T['Technology'] == technology].T

    @staticmethod
    def filter_cec_inverters_by_modules(
        modules: pd.DataFrame, 
        hot_temperature: float, 
        cold_temperature: float,
        voc_margin=0.97
        ) -> pd.DataFrame:
        inverters = pv.retrieve_sam('cecinverter')
        filtered_inverters = {}
        for _, mp in modules.items():

            Vmp_hot, Voc_hot_dummy = PVSystemHelper.module_iv_at(mp, hot_temperature)
            Vmp_cold, Voc_cold = PVSystemHelper.module_iv_at(mp, cold_temperature)

            for iname, inv in inverters.items():

                mppt_low = float(inv['Mppt_low'])
                mppt_high = float(inv['Mppt_high'])
                vdcmax = float(inv['Vdcmax'])

                n_min = int(np.ceil (mppt_low  / max(Vmp_hot,  1e-6)))
                n_max_mppt = int(np.floor(mppt_high / max(Vmp_cold, 1e-6)))
                n_max_voc  = int(np.floor(voc_margin * vdcmax / max(Voc_cold, 1e-6)))
                n_max = min(n_max_mppt, n_max_voc)
                if n_min <= n_max and n_max >= 1:
                    filtered_inverters[iname] = inv
        return pd.DataFrame(filtered_inverters)

    @staticmethod
    def module_iv_at(mp, Tcell, poa=1000):
        calc = pv.calcparams_cec(
            effective_irradiance=poa, temp_cell=Tcell,
            alpha_sc=float(mp['alpha_sc']), a_ref=float(mp['a_ref']),
            I_L_ref=float(mp['I_L_ref']), I_o_ref=float(mp['I_o_ref']),
            R_sh_ref=float(mp['R_sh_ref']), R_s=float(mp['R_s']),
            Adjust=float(mp.get('Adjust', 1.0)),
            EgRef=float(mp.get('EgRef', 1.121)), dEgdT=float(mp.get('dEgdT', -0.0002677)),
        )
        sd = pv.singlediode(*calc, method='lambertw')
        return float(sd['v_mp']), float(sd['v_oc'])