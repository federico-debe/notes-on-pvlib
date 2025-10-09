import numpy as np
import pandas as pd

from pvlib import pvsystem as pv

from models.pv_custom.enums import RackingType, TrackingType


class PVSystemHelper:
    _CACHED_INVERTERS = None
    _CACHED_INVERTER_DATA = None


    def filter_cec_inverters_by_kwp(self, kwp, oversize_ratio=1.3):
        inverters = pv.retrieve_sam('cecinverter')
        if kwp <= 15:
            max_number_of_inverters = 1
            max_inverter_power_w = 20000.0
        elif kwp <= 100:
            max_number_of_inverters = 2
            max_inverter_power_w = 110000.0
        elif kwp <= 500:
            max_number_of_inverters = max(2, int(kwp / 50))
            max_inverter_power_w = 300000.0
        else:
            max_number_of_inverters = max(4, int(kwp / 100))
            max_inverter_power_w = 500000.0
        system_size_w = kwp * 1000
        min_inverter_power = system_size_w / (oversize_ratio * max_number_of_inverters)

        power_filter = (
            (inverters.T['Paco'] >= min_inverter_power) & 
            (inverters.T['Paco'] <= max_inverter_power_w)
        )
        filtered_inverters = inverters.T[power_filter].T
        return filtered_inverters

    def filter_cec_modules_by_tech_choice(self, technology) -> pd.DataFrame:
        cecmodules = pv.retrieve_sam('cecmod')
        tech_modules = cecmodules.T[cecmodules.T['Technology'] == technology].T

        return tech_modules

    @staticmethod
    def filter_cec_inverters_by_modules(
        modules: pd.DataFrame, 
        hot_temperature: float, 
        cold_temperature: float,
        voc_margin=0.97
    ) -> pd.DataFrame:
        
        # Get inverters once
        inverters = pv.retrieve_sam('cecinverter')
        
        # Convert inverters to DataFrame for vectorization
        inv_df = pd.DataFrame(inverters).T
        inv_df['Mppt_low'] = inv_df['Mppt_low'].astype(float)
        inv_df['Mppt_high'] = inv_df['Mppt_high'].astype(float) 
        inv_df['Vdcmax'] = inv_df['Vdcmax'].astype(float)
        
        compatible_inverters = set()
        
        for _, mp in modules.items():
            Vmp_hot, Voc_hot_dummy = PVSystemHelper.module_iv_at(mp, hot_temperature)
            Vmp_cold, Voc_cold = PVSystemHelper.module_iv_at(mp, cold_temperature)
            
            if Vmp_hot <= 0 or Vmp_cold <= 0 or Voc_cold <= 0:
                continue
            
            # Vectorized operations on the entire inverter DataFrame
            n_min = np.ceil(inv_df['Mppt_low'] / Vmp_hot).astype(int)
            n_max_mppt = np.floor(inv_df['Mppt_high'] / Vmp_cold).astype(int)
            n_max_voc = np.floor(voc_margin * inv_df['Vdcmax'] / Voc_cold).astype(int)
            n_max = np.minimum(n_max_mppt, n_max_voc)
            
            # Filter compatible inverters
            compatible_mask = (n_min <= n_max) & (n_max >= 1)
            compatible_inv_names = inv_df[compatible_mask].index
            compatible_inverters.update(compatible_inv_names)
        
        return pd.DataFrame({name: inverters[name] for name in compatible_inverters})

    @staticmethod
    def filter_cec_modules_by_inverters(
        modules: pd.DataFrame,
        inverters: pd.DataFrame,
        hot_temperature: float, 
        cold_temperature: float,
        voc_margin=0.97,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Batched vectorized processing for very large module databases
        """
        # Convert inverters to arrays once
        inv_df = pd.DataFrame(inverters).T
        mppt_lows = inv_df['Mppt_low'].astype(float).values
        mppt_highs = inv_df['Mppt_high'].astype(float).values
        vdcmaxs = inv_df['Vdcmax'].astype(float).values
        
        n_inverters = len(mppt_lows)
        compatible_modules = {}
        
        # Process modules in batches to manage memory
        module_items = list(modules.items())
        
        for i in range(0, len(module_items), batch_size):
            batch = module_items[i:i + batch_size]
            batch_module_names = []
            batch_Vmp_hot = []
            batch_Vmp_cold = [] 
            batch_Voc_cold = []
            
            # Calculate module parameters for this batch
            for module_name, mp in batch:
                Vmp_hot, Voc_hot_dummy = PVSystemHelper.module_iv_at(mp, hot_temperature)
                Vmp_cold, Voc_cold = PVSystemHelper.module_iv_at(mp, cold_temperature)
                
                if Vmp_hot > 1e-6 and Vmp_cold > 1e-6 and Voc_cold > 1e-6:
                    batch_module_names.append(module_name)
                    batch_Vmp_hot.append(Vmp_hot)
                    batch_Vmp_cold.append(Vmp_cold)
                    batch_Voc_cold.append(Voc_cold)
            
            if not batch_module_names:
                continue
                
            # Convert to arrays
            Vmp_hot_arr = np.array(batch_Vmp_hot)
            Vmp_cold_arr = np.array(batch_Vmp_cold)
            Voc_cold_arr = np.array(batch_Voc_cold)
            n_batch_modules = len(Vmp_hot_arr)
            
            # Vectorized broadcasting
            Vmp_hot_2d = Vmp_hot_arr[:, np.newaxis]
            Vmp_cold_2d = Vmp_cold_arr[:, np.newaxis]
            Voc_cold_2d = Voc_cold_arr[:, np.newaxis]
            
            mppt_lows_2d = mppt_lows[np.newaxis, :]
            mppt_highs_2d = mppt_highs[np.newaxis, :]
            vdcmaxs_2d = vdcmaxs[np.newaxis, :]
            
            # Vectorized calculations
            with np.errstate(divide='ignore', invalid='ignore'):
                n_min = np.ceil(mppt_lows_2d / Vmp_hot_2d).astype(int)
                n_max_mppt = np.floor(mppt_highs_2d / Vmp_cold_2d).astype(int)
                n_max_voc = np.floor(voc_margin * vdcmaxs_2d / Voc_cold_2d).astype(int)
            
            n_max = np.minimum(n_max_mppt, n_max_voc)
            compatibility_matrix = (n_min <= n_max) & (n_max >= 1)
            batch_compatible = np.any(compatibility_matrix, axis=1)
            
            # Add compatible modules from this batch
            for j, is_compatible in enumerate(batch_compatible):
                if is_compatible:
                    module_name = batch_module_names[j]
                    compatible_modules[module_name] = modules[module_name]
        
        print(f"Batched module filter: {len(modules)} -> {len(compatible_modules)} modules")
        return pd.DataFrame(compatible_modules)

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