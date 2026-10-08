from scipy.stats import norm
# from scipy.stats.Normal import pdf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from config import fast_test
from system.utils.SimulationCache import SimulationCache
from main import gen_initial_arch, get_calib_cost_avg, calculate_cost, set_workload
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

SYS_ARRAY = {
    "64x64":64,
    "96x96":96,
    "128x128":128,
    "192x192":192
}

TECH_NODES = {
    "7":7,
    "10":10,
    "14":14
}

INTER_PKG_ARCH = {
    "2d_na":[0,0,0,0,0,1],
    "2.5d_emib":[0,0,0,0,1,0],
    "2.5d_rdl":[0,0,0,1,0,0],
    "3d_tsv":[0,0,1,0,0,0],
    "3d_u_bump":[0,1,0,0,0,0],
    "3d_hyb_bond":[1,0,0,0,0,0]
}

PKG_PROTOCOL = {
    "ucie_std":[0,0,0,0,1],
    "ucie_adv":[0,0,0,1,0],
    "aib":[0,0,1,0,0],
    "bow":[0,1,0,0,0],
    "ucie_3d":[1,0,0,0,0]
}

PKG_MEM_ARCH = {
    "ddr4":[0,0,0,1],
    "ddr5":[0,0,1,0],
    "hbm2":[0,1,0,0],
    "hbm3":[1,0,0,0]
}

DATAFLOW = {
    "ws":[0,0,1],
    "os":[0,1,0],
    "is":[1,0,0]
}

def architecture_to_vector(arch, max_chiplets=6):
    '''
    Converts the architecture to a vector.
    The vector has this format (values at particular indices):
    0: number of chiplets,
    1-3: tech node, array size and sram buffer for chiplet 1,
    4-6: tech node, array size and sram buffer for chiplet 2,
    7-9: tech node, array size and sram buffer for chiplet 3,
    10-12: tech node, array size and sram buffer for chiplet 4,
    13-15: tech node, array size and sram buffer for chiplet 5,
    16-18: tech node, array size and sram buffer for chiplet 6,
    19-24: inter packaging architecture,
    25-29: packaging protocol,
    30-33: packaging memory architecture,
    34: split-K, 
    35: assign ascending, 
    36-38: dataflow.
    '''
    if not isinstance(arch, dict):
        raise TypeError("arch must be dictionary")
    vector = []
    chiplets = [k for k in arch.keys() if "Chiplet_" in k]
    num_chiplets = len(chiplets)
    vector.append(num_chiplets)
    for i in range(max_chiplets):
        if i < num_chiplets:
            vector.extend([
                TECH_NODES[arch["Chiplet_"+str(i+1)]["tech_node"]],
                SYS_ARRAY[arch["Chiplet_"+str(i+1)]["sys_array_size"]],
                arch["Chiplet_"+str(i+1)]["sram_buf"]
                ])
        else:
            vector.extend([0,0,0])
    pkg = arch["pkg"]
    # vector.extend(INTER_PKG_ARCH[pkg["inter_pkg_conn"][0]["connection_type"]])
    # if pkg["protocol_3d"] == "na" and pkg["protocol_2.5d"] == "na":
    #     vector.extend([0,0,0,0,0])
    # elif pkg["protocol_3d"] != "na":
    #     vector.extend(PKG_PROTOCOL[pkg["protocol_3d"]])
    # else:
    #     vector.extend(PKG_PROTOCOL[pkg["protocol_2.5d"]])
    types = {c["connection_type"] for c in pkg["inter_pkg_conn"]} or {"2d_na"}
    vector.extend([int(any(INTER_PKG_ARCH[t][i] for t in types)) for i in range(6)])
    protos = {pkg["protocol_3d"], pkg["protocol_2.5d"]} -{"na"}
    vector.extend([int(any(PKG_PROTOCOL[p][i] for p in protos)) for i in range(5)])
    vector.extend(PKG_MEM_ARCH[pkg["mem_pkg_conn"]["mem_type"]])
    m = arch["WL_mapping"]["mapping"]
    vector.append(m["if_splitting_k"])
    vector.append(m["assign_workload_in_ascending_order"])
    vector.extend(DATAFLOW[m["dataflow"][0]])
    # sc = StandardScaler().fit(vector)
    # vector = sc.transform(vector)
    return vector

def expected_improvement(X, gp, best_y, epsilon=0.01):
    mean, std = gp.predict(X, return_std=True)
    z = (best_y - mean - epsilon) / std
    ei = std * (z * norm.cdf(x=z) + norm.pdf(x=z))
    return ei

class BayesianOptimizer:
    def __init__(self, config_path, wl_idx, cache_file, run_name, cost_profile="t1", max_chiplets=6, random_seed=42):
        self.config_path, self.max_chiplets, self.wl_idx, self.cost_profile = config_path, max_chiplets, wl_idx, cost_profile
        self.cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
        k = ConstantKernel() * Matern(np.ones(39), nu=2.5) + WhiteKernel(1e-2)
        self.gp = GaussianProcessRegressor(kernel=k, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=True, random_state=random_seed)
        self.X = []
        self.y = []
        self.architectures = []
        self.txt = ''

    def evaluate(self, cur_architecture):
        cost_val, norm_cost_dict, raw_cost_dict = calculate_cost(
                profile_name=self.cost_profile,
                cost_avgerage=self.cost_avg,
                system_dict=cur_architecture,
                cache=self.cache
                )
        return cost_val, norm_cost_dict, raw_cost_dict

    def initialize(self, n_initial):
        calibration_iterations = 500
        calibration_file_path = f"cfg/calibration/calibration_{self.wl_idx}.json"
        self.cost_avg = get_calib_cost_avg(
                    calibration_iterations=calibration_iterations,
                    config_path=self.config_path,
                    cache=self.cache,
                    calibration_file_path=calibration_file_path 
                )
        for i in range(n_initial):
            arch = gen_initial_arch(config_path=self.config_path,stack_diff_size=True)
            if arch is None:
                continue
            x = architecture_to_vector(arch)
            y, _, _ = self.evaluate(arch)
            self.X.append(x)
            self.y.append(y)
            self.architectures.append(arch)
            print(f"[BO INIT] {i+1}/{n_initial} cost: {y}")
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        self.sc = StandardScaler().fit(self.X)
        self.X = self.sc.transform(self.X)

    def get_candidates(self, num_candidates):
        candidates = []
        for _ in range(num_candidates):
            arch = gen_initial_arch(config_path=self.config_path,stack_diff_size=True)
            x = architecture_to_vector(arch)
            candidates.append((x, arch))
        return candidates

    def select_next(self, num_candidates=200):
        self.gp.fit(self.X, self.y)
        arch_candidates = self.get_candidates(num_candidates)
        best_y = np.min(self.y)
        candidates = [x for (x,_) in arch_candidates]
        candidates = self.sc.transform(candidates)
        ei = expected_improvement(candidates, self.gp, best_y)
        best_idx = np.argmax(ei)
        return candidates[best_idx], ei[best_idx], arch_candidates[best_idx][1]

    def step(self, num_candidates):
        candidate, ei, arch = self.select_next(num_candidates=num_candidates)
        cost, _, _= self.evaluate(arch)
        self.X = np.vstack([self.X, candidate])
        self.y = np.append(self.y, cost)
        self.architectures.append(arch)
        best_idx = np.argmin(self.y)
        print(f"Best cost:{self.y[best_idx]}\nNew cost:{cost}\nEI:{ei}")
        self.txt += f"Best cost:{str(self.y[best_idx])}\nNew cost:{str(cost)}\nEI:{str(ei)}\n"

    def run(self, n_iterations, num_candidates):
        self.initialize(n_initial=50)
        for i in range(n_iterations):
            self.step(num_candidates)
            print(f"Iteration {i+1}/{n_iterations}")
            self.txt += f"Iteration {str(i+1)}/{str(n_iterations)}\n"
            if i//50 == 0:
                print("Kernel", self.gp.kernel_)
                self.txt += "Kernel: "+str(self.gp.kernel_)+"\n"
        best_idx = np.argmin(self.y)
        print(f"For workload:{self.wl_idx}")
        self.txt += f"For workload:{self.wl_idx}\n"
        print(f"Final minimum cost:{self.y[best_idx]}\nBest architecture:{self.architectures[best_idx]}\n")
        self.txt += f"Final minimum cost:{self.y[best_idx]}\nBest architecture:{self.architectures[best_idx]}\n"
        np.savez("bayes_opt_runs/"+self.wl_idx, X=self.X, y=self.y, architectures=np.array(self.architectures, dtype=object))
        print("Saved run:", self.wl_idx)
        self.txt += "Saved run:"+self.wl_idx+"\n"
        self.X = list(self.X)
        self.y = list(self.y)
        print("Length of X", len(self.X))
        print("Length of y", len(self.y))
        return self.X[best_idx], self.y[best_idx], self.architectures[best_idx]

if __name__=="__main__":
    input_file_path = "cfg/parameters/input.json"
    cache_file = "cfg/static_cache/static_cache.csv"
    run_name = "bayes_opt_try"
    wl_name = "workload_transformer"
    wl_id = "1"
    start = time.time()
    set_workload(wl_name,wl_id)
    bo = BayesianOptimizer(config_path=input_file_path, wl_idx=f"{wl_name}_{wl_id}", cache_file=cache_file, run_name=run_name)
    _, best_cost, best_arch = bo.run(n_iterations=500, num_candidates=500)
    print("Time taken for this workload:", time.time()-start)
    with open(f"bayes_opt_runs/prints_{wl_name}_{wl_id}.txt", "w") as f:
        f.write(bo.txt)