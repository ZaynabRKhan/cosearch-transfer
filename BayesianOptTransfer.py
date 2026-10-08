from BayesianOptimization import *

class BayesianOptTransfer:
    def __init__(self, config_path, wl_idx, cache_file, run_name, old_wl_idx, cost_profile="t1", max_chiplets=6, random_seed=42):
        self.config_path, self.max_chiplets, self.wl_idx, self.cost_profile, self.old_wl_idx = config_path, max_chiplets, wl_idx, cost_profile, old_wl_idx
        self.cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
        k = ConstantKernel() * Matern(np.ones(39), nu=2.5) + WhiteKernel(1e-2)
        self.gp = GaussianProcessRegressor(kernel=k, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=True, random_state=random_seed)
        # self.X = []
        # self.y = []
        # self.architectures = []
        data = np.load("bayes_opt_runs/"+self.old_wl_idx+".npz", allow_pickle=True)
        self.X = data["X"]
        self.y = data["y"]
        self.architectures = data["architectures"]
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
        y_idx = np.argsort(self.y)
        # y_idx = y_idx[:n_initial]
        # self.X = self.X[y_idx]
        # self.y = self.y[y_idx]
        # self.architectures = self.architectures[y_idx]
        Xs, ys, archs, seen = [], [], [], set()
        for id in y_idx:
            a = self.architectures[id]
            x = architecture_to_vector(a)
            if tuple(x) in seen:
                continue
            seen.add(tuple(x))
            y, _, _ = self.evaluate(a)
            Xs.append(x)
            ys.append(y)
            archs.append(a)
            if len(seen) == n_initial:
                break
        self.X = Xs
        self.y = ys
        self.architectures = archs
        self.sc = StandardScaler().fit(self.X)
        self.X = self.sc.transform(self.X)
            
        # for i in range(n_initial):
        #     arch = gen_initial_arch(config_path=self.config_path,stack_diff_size=True)
        #     if arch is None:
        #         continue
        #     x = architecture_to_vector(arch)
        #     y, _, _ = self.evaluate(arch)
        #     self.X.append(x)
        #     self.y.append(y)
        #     self.architectures.append(arch)
        #     print(f"[BO INIT] {i+1}/{n_initial} cost: {y}")
        # self.X = np.asarray(self.X)
        # self.y = np.asarray(self.y)

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
        self.txt += f"Best cost:{self.y[best_idx]}\nNew cost:{cost}\nEI:{ei}\n"

    def run(self, n_iterations, num_candidates):
        self.initialize(n_initial=50)
        for i in range(n_iterations):
            self.step(num_candidates)
            print(f"Iteration {i+1}/{n_iterations}")
            self.txt += f"Iteration {i+1}/{n_iterations}\n"
            if i//50 == 0:
                print("Kernel", self.gp.kernel_)
                self.txt += "Kernel: "+str(self.gp.kernel_)+"\n"
        best_idx = np.argmin(self.y)
        print(f"For workload:{self.wl_idx}")
        self.txt += f"For workload:{self.wl_idx}\n"
        print(f"Final minimum cost:{self.y[best_idx]}\nBest architecture:{self.architectures[best_idx]}")
        self.txt += f"Final minimum cost:{self.y[best_idx]}\nBest architecture:{self.architectures[best_idx]}\n"
        # print("Transfer:", transfer)
        
        # if transfer==True:
        np.savez("bayes_opt_runs/"+self.old_wl_idx+"_"+self.wl_idx, X=self.X, y=self.y, architectures=np.array(self.architectures, dtype=object))
        print("Saved run:", self.old_wl_idx+"_"+self.wl_idx)
        self.txt += "Saved run:"+self.old_wl_idx+"_"+self.wl_idx+"\n"
        # else:
            # np.savez("bayes_opt_runs/test_"+self.wl_idx, X=self.X, y=self.y, architectures=np.array(self.architectures, dtype=object))
            # print("Didn't save run:", self.wl_idx)
        self.X = list(self.X)
        self.y = list(self.y)
        print("Length of X", len(self.X))
        print("Length of y", len(self.y))
        return self.X[best_idx], self.y[best_idx], self.architectures[best_idx]

    # def transfer_initialize(self):

    # def transfer_run(self, n_iterations, num_candidates, t_wl_idx):
    #     self.wl_idx = t_wl_idx
    #     wl_id = t_wl_idx.split("_")[-1]
    #     wl_name = t_wl_idx.split("_")[0] + "_" + t_wl_idx.split("_")[1]
    #     print("wl id:", wl_id)
    #     print("wl name",wl_name)
    #     set_workload(wl_name,wl_id)
    #     self.initialize(n_initial=150)
    #     print(f"Continuing search by transfer from {self.wl_idx} to {t_wl_idx}")
    #     for i in range(n_iterations):
    #         self.step(num_candidates)
    #         print(f"Iteration {i+1}/{n_iterations}")
    #     best_idx = np.argmin(self.y)
    #     print(f"For workload from: {self.wl_idx} to {t_wl_idx}")
    #     print(f"Final minimum cost:{self.y[best_idx]}\nBest architecture:{self.architectures[best_idx]}")
    #     np.savez("bayes_opt_runs/"+self.wl_idx+"_"+t_wl_idx, X=self.X, y=self.y, architectures=np.array(self.architectures, dtype=object))
    #     print("Saved run:", t_wl_idx)
    #     # best_X, best_cost, best_arch = self.run(n_iterations, num_candidates)
    #     print("Length of X", np.size(self.X))
    #     print("Length of y", np.size(self.y))
    #     return self.X[best_idx], self.y[best_idx], self.architectures[best_idx]

if __name__=="__main__":
    input_file_path = "cfg/parameters/input.json"
    cache_file = "cfg/static_cache/static_cache.csv"
    run_name = "bayes_opt_try"
    old_wl_idx = "workload_cnn_1"
    wl_name = "workload_transformer"
    wl_id = "1"
    start = time.time()
    set_workload(wl_name,wl_id)
    bo = BayesianOptTransfer(config_path=input_file_path, wl_idx=f"{wl_name}_{wl_id}", cache_file=cache_file, run_name=run_name, old_wl_idx=old_wl_idx)
    _, best_cost, best_arch = bo.run(n_iterations=500, num_candidates=500)
    print("Time taken for this workload:", time.time()-start)
    with open(f"bayes_opt_runs/prints_{old_wl_idx}_{wl_name}_{wl_id}.txt", "w") as f:
        f.write(bo.txt)
