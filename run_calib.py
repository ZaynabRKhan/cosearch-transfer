from main import get_calib_cost_avg, gen_initial_arch
from system.utils.SimulationCache import SimulationCache
from config import fast_test

print("Running calibration...")

calibration_iterations = 100
input_file_path = "cfg/parameters/input.json"
calibration_file_path = "cfg/calibration/calibration_saved.json"
cache_file = "cfg/static_cache/static_cache.csv"
run_name = "cost_calulation_only"

cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)

final_config = gen_initial_arch(config_path=input_file_path)

get_calib_cost_avg(
    calibration_iterations=calibration_iterations, 
    config_path=input_file_path, 
    calibration_file_path=calibration_file_path,
    cache=cache
    )

