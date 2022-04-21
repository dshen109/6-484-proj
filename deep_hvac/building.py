from rc_simulator import supply_system
from rc_simulator.building_physics import Zone
from rc_simulator.emission_system import AirConditioning
from rc_simulator.radiation import Window


AZIMUTH_SOUTH = 0
AZIMUTH_WEST = 90
AZIMUTH_NORTH = 180
AZIMUTH_EAST = 270

OFFICE_OCCUPANCY = 200  # ft**2/person

WINDOW_AREA_NORTH = 196
WINDOW_AREA_EAST = 131
WINDOW_AREA_SOUTH = WINDOW_AREA_NORTH
WINDOW_AREA_WEST = WINDOW_AREA_EAST

SHGC = 0.251

HOUSTON_LAT = 29.749907
HOUSTON_LONG = -95.358421

OCCUPIED_HEATING_STPT = 20  # 68F
OCCUPIED_COOLING_STPT = 25.56  # 78F

UNOCCUPIED_HEATING_STPT = 18.33  # 65F
UNOCCUPIED_COOLING_STPT = 29.44  # 85F


def default_building():
    """Make an office building representative of a default building.

    It is intended to emulate a DOE reference medium office in zone 2A
    (Houston), post 1980
    """
    # There are 3 floors
    single_floor_area = 3321  # m**2
    gross_wall_area = 1978  # m**2
    volume = 19741  # m**3
    outside_air_req = 0.1  # cubic ft / min / ft**2
    infiltration = 0.001133  # m**3 / s / m**2

    ach_vent = outside_air_req * 10.76 * 1.699 / volume
    ach_infil = infiltration * gross_wall_area * 3600 / volume

    n_occupants = int(10.76 * (3 * single_floor_area) / OFFICE_OCCUPANCY)

    params = dict(
        window_area=653,
        walls_area=gross_wall_area,
        floor_area=single_floor_area * 2,
        room_vol=volume,
        total_internal_area=(6 * single_floor_area + gross_wall_area),
        u_walls=0.852,
        u_windows=1.22,
        ach_vent=ach_vent,
        ach_infl=ach_infil,
        ventilation_efficiency=0,
        thermal_capacitance_per_floor_area=165000,
        t_set_heating=OCCUPIED_HEATING_STPT,
        t_set_cooling=OCCUPIED_COOLING_STPT,
        max_cooling_energy_per_floor_area=-464338 / single_floor_area / 3,
        max_heating_energy_per_floor_area=220516 / single_floor_area / 3,
        heating_supply_system=supply_system.HeatPumpAir,
        cooling_supply_system=supply_system.HeatPumpAir,
        heating_emission_system=AirConditioning,
        cooling_emission_system=AirConditioning
    )

    zone = Zone(**params)

    zone.occupants = n_occupants

    windows = [
        Window(AZIMUTH_NORTH, glass_solar_transmittance=SHGC,
               area=WINDOW_AREA_NORTH),
        Window(AZIMUTH_SOUTH, glass_solar_transmittance=SHGC,
               area=WINDOW_AREA_SOUTH),
        Window(AZIMUTH_EAST, glass_solar_transmittance=SHGC,
               area=WINDOW_AREA_EAST),
        Window(AZIMUTH_WEST, glass_solar_transmittance=SHGC,
               area=WINDOW_AREA_WEST)
    ]

    return zone, windows, HOUSTON_LAT, HOUSTON_LONG


def comfort_temperature(t_outdoor):
    return 17.8 + 0.31 * t_outdoor
