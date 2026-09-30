from main import calculate_cost, get_calib_cost_avg, gen_initial_arch
from system.utils.SimulationCache import SimulationCache
from config import fast_test
import pprint, json

print("Running Cost Calculation only")

pp = pprint.PrettyPrinter(indent=4, width=50)
calibration_iterations = 100
calibration_file_path = "cfg/calibration/calibration_saved.json"
cache_file = "cfg/static_cache/static_cache.csv"
run_name = "cost_calulation_only"
cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)

def get_arch_cost():
    input_file_path = "cfg/parameters/input.json"
    final_config = gen_initial_arch(config_path=input_file_path)
    # with open("cfg/parameters/input.json") as f:
    #     final_config = json.load(f)

    print("REturned architecture:")
    pp.pprint(final_config)
    print(type(final_config))

    cost_avg = get_calib_cost_avg(
        calibration_iterations=calibration_iterations, 
        config_path=input_file_path, 
        calibration_file_path=calibration_file_path,
        cache=cache
        )

    cost_val, norm_cost_dict, raw_cost_dict = calculate_cost(
        profile_name='t1',
        cost_avgerage=cost_avg,
        system_dict=final_config,
        cache=cache
        )
    print("cost_val,:", cost_val)
    print("norm_cost_dict")
    pp.pprint(norm_cost_dict)
    print("raw_cost_dict:")
    pp,pprint(raw_cost_dict)
    return cost_val, norm_cost_dict, raw_cost_dict

if __name__=="__main__":
    get_arch_cost()