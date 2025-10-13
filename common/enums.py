from enum import Enum
from pvlib.pvsystem import FixedMount, SingleAxisTrackerMount

from models.simulatorrecs.dual_axis_tracker_mount import DualAxisTrackerMount


class RackingType(Enum):
    '''
    open rack       <-> free standing               --> coolest
    close mount     <-> close to roof               --> warmer
    insulated back  <-> integrated into building    --> hottest
    '''
    open_rack = 0
    close_mount = 1
    insulated_back = 2
    freestanding = 3
    insulated = 4
    semi_integrated = 5

    @staticmethod
    def get_default_model_temperature_params(racking_type):
        u_v = 0
        u_c = 0
        if racking_type.value == RackingType.open_rack.value or racking_type.value == RackingType.freestanding.value:
            u_v = 29.0
        elif racking_type.value == RackingType.close_mount.value or racking_type.value == RackingType.semi_integrated.value:
            u_v = 20.0
        else:
            u_v = 15.0
        return u_v, u_c

class WiringMaterial(Enum):
    copper = 0
    aluminum = 1

    @staticmethod
    def _rho20(material) -> float:
        '''ohm·m @20°C'''
        if material.value == WiringMaterial.copper.value:
            return 1.724e-8
        elif material.value == WiringMaterial.aluminum.value:
            return 2.826e-8
        return 1.724e-8 if material.value == WiringMaterial.copper.value else 2.826e-8

    @staticmethod
    def _alpha_T(material) -> float:
        '''1/°C'''
        if material.value == WiringMaterial.copper.value:
            return 0.00393
        elif material.value == WiringMaterial.aluminum.value: 
            return 0.00403
        raise Exception('Unknown material')

class TechChoice(Enum):
    CIGS = 0
    CdTe = 1
    Mono_c_Si = 2
    Multi_c_Si = 3
    Thin_Film = 4

    @staticmethod
    def get_tech_choice_descritpion(tech_choice):
        if tech_choice.value == TechChoice.CIGS.value:
            return 'CIGS'
        elif tech_choice.value == TechChoice.CdTe.value:
            return 'CdTe'
        elif tech_choice.value == TechChoice.Mono_c_Si.value:
            return 'Mono-c-Si'
        elif tech_choice.value == TechChoice.Multi_c_Si.value:
            return 'Multi-c-Si'
        else:
            return 'Thin Film'

class TrackingType(Enum):
    '''type of tracking for pv plant'''
    fixed = 0
    single_horizontal_axis_aligned_north_south = 1
    two_axis_tracking = 2
    vertical_axis_tracking = 3
    single_horizontal_axis_aligned_east_west = 4
    single_inclined_axis_aligned_north_south = 5

    @staticmethod
    def get_mount(
        tracking_type, 
        racking_model: RackingType, 
        tilt=0.0, 
        azimuth=0.0, 
        max_angle=0.0, 
        gcr=0.35, 
        backtrack=True
        ):
        if tracking_type.value == TrackingType.fixed.value:
            return FixedMount(
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                racking_model=racking_model.name,
            )
        elif tracking_type.value == TrackingType.single_horizontal_axis_aligned_north_south.value:
            return SingleAxisTrackerMount(
                axis_tilt=0.0,          # horizontal axis
                axis_azimuth=180.0,     # N–S axis pointing south (0.0 also ok)
                max_angle=max_angle,    # mechanical limit
                backtrack=backtrack,
                gcr=gcr,
                racking_model=racking_model.name,
            )
        elif tracking_type.value == TrackingType.vertical_axis_tracking.value:
            return SingleAxisTrackerMount(
                axis_tilt=90.0,         # vertical rotation axis
                axis_azimuth=0.0,       # arbitrary for vertical; 0 is fine
                max_angle=180.0,
                backtrack=False,        # no backtracking for vertical axis
                gcr=1.0,                # ignored here
                racking_model=racking_model.name,
            )
        elif tracking_type.value == TrackingType.single_horizontal_axis_aligned_east_west.value:
            return SingleAxisTrackerMount(
                axis_tilt=0.0,
                axis_azimuth=90.0,      # E–W axis
                max_angle=max_angle,
                backtrack=True,
                gcr=gcr,
                racking_model=racking_model.name,
            )
        elif tracking_type.value == TrackingType.single_inclined_axis_aligned_north_south.value:
            return SingleAxisTrackerMount(
                axis_tilt=tilt,          # e.g., site slope or design (5–20° typical)
                axis_azimuth=180.0,      # N–S axis
                max_angle=max_angle,
                backtrack=backtrack,
                gcr=gcr,
                racking_model=racking_model.name,
            )
        else: # two axis tracking
            return DualAxisTrackerMount()


class SurfaceType(Enum):
    urban = 0
    grass = 1
    fresh_grass = 2
    soil = 3
    sand = 4
    snow = 5
    fresh_snow = 6
    asphalt = 7
    concrete = 8
    aluminum = 9
    copper = 10
    fresh_steel = 11
    dirty_steel = 12
    sea = 13


class EditStatus(Enum):
    NONE = 0
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    UPDATE_AFTER_OPTIMIZATION = 4

class EnergyPlantType(Enum):
    PV = 0
    GEOTHERMAL = 1
    NONE = -100

class PVBatteryConnectionType(Enum):
    NONE = 0
    DC = 1 # PV → Battery
    AC = 2  # PV → Inverter → Battery

class PriceType(Enum):
    ENERGY_PRICE = 0
    TECHNOLOGY_PRICE = 1
    ENERGY_PURCHASE_TYPE = 2
    #ENERGY_SELLING_TYPE = 3
    INCENTIVE = 4
    GENERIC = 5
    TAX = 6

class SortOption(Enum):
    ALL = 0
    PCC_ASC = 1
    PCC_DESC = 2
    NAME_ASC = 3
    NAME_DESC = 4
    CONTRACT_POWER_ASC = 5
    CONTRACT_POWER_DESC = 6
    YEARLY_CONSUMPTION_ASC = 7
    YEARLY_CONSUMPTION_DESC = 8
    REGION = 9
    STATUS = 10
    TYPE = 11
    CREATION_DATE_ASC = 12
    CREATION_DATE_DESC = 13