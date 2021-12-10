import sys, logging
import time
import numpy as np


from utils.binpackingsolution import BinPackingSolutions
from utils.binpackingsolution import BinDetails

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s: %(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class FSO:
    def __init__(self, male_bugs, female_bugs, s1_max, s2_max):
        self.male_bugs = male_bugs
        self.female_bugs = female_bugs
        self.total_bugs = male_bugs * female_bugs
        self.s1_max = s1_max
        self.s2_max = s2_max
        self.L1 = 34
        self.L2 = 11
        self.bins_data = BinPackingSolutions()
        self.min_fitness_ids = []
        self.rng = np.random.default_rng(98765)

    def __repr__(self):
        return f"FSO (Male_bugs = {self.male_bugs}, Female_bugs = {self.female_bugs})"
    
    def __str__(self):
        return f"FSO (Male_bugs = {self.male_bugs}, Female_bugs = {self.female_bugs})"

    @property
    def red_male_bugs(self) -> np.ndarray:
        return self._red_male_bugs

    @red_male_bugs.setter
    def red_male_bugs(self, red_male_bugs: np.ndarray):
        self._red_male_bugs = red_male_bugs

    def find_score(self, bin_sol: BinDetails=None):
        """Find score using the formula:
            score = 1 - (sum((sum_of_weight[i]/c)^2 for each bin i)/no_of_bins)
        """
        k = 2
        no_of_bins = bin_sol.no_of_bins
        bin_sum = self.bins_data.max_bin_capacity - np.array(bin_sol.free_bin_caps)
        score = 1 - (np.sum((bin_sum/self.bins_data.max_bin_capacity) ** k)/no_of_bins)
        bin_sol.score = score


    def generate_bin_solution(self, new_bug=None, initial=False, final_best=False)-> BinDetails:
        if initial:
            bin_sol = self.bins_data.best_fit_heuristic()
        elif final_best:
            bin_sol = self.bins_data.best_fit_algorithm(self.glo_best_rov)
        else:
            bin_sol = self.bins_data.best_fit_algorithm(new_bug)
        self.find_score(bin_sol)
        return bin_sol

    def generate_init_population(self):
        """Generation of initial population
        """
        logger.info(f"Generating initial population (Number of frogs: {self.total_bugs})")
        
        red_male_bugs = np.empty((self.male_bugs, self.bins_data.no_of_items, self.female_bugs))
        min_score_ids = [-1] * self.male_bugs
        global_min_score = 1
        for mb in range(self.male_bugs):
            min_score = 1
            rov_each_bin = []
            for fb in range(self.female_bugs):
                res_bin = self.generate_bin_solution(initial=True)
                rov_each_bin.append(res_bin.rov_continous)
                if min_score > res_bin.score:
                    min_score_ids[mb] = fb
                    min_score = res_bin.score
            if global_min_score > min_score:
                global_min_score = min_score
                glo_best_rov = rov_each_bin[min_score_ids[mb]]
            
            red_male_bugs[mb] = np.array(rov_each_bin).T
        
        self.min_fitness_ids = min_score_ids
        self.red_male_bugs = red_male_bugs
        self.global_min_score = global_min_score
        self.glo_best_rov = glo_best_rov

    def main_search(self):
        for s1 in range(self.s1_max):
            for j in range(self.L1):
                self.bins_data.no_of_items
                a = self.rng.permutation(self.male_bugs)
                for m in range(self.male_bugs):
                    # logger.info(f"Iteration S1: {s1} -- L1: {j}: red_bug {m} STARTED")
                    c1 = -0.75 + 2.255 * self.rng.random((self.bins_data.no_of_items, self.female_bugs))
                    c2 = -0.25 + 1.302 * self.rng.random((self.bins_data.no_of_items, self.female_bugs))
                    b = a[m];

                    male = self.red_male_bugs[m, :, self.min_fitness_ids[m]]
                    male1 = self.red_male_bugs[b, :, self.min_fitness_ids[b]]
                    male = np.transpose([male] * self.female_bugs)
                    male1 = np.transpose([male1] * self.female_bugs)

                    new_female_bugs = self.red_male_bugs[m] + c1 * (male - self.red_male_bugs[m]) + c2 * (male1 - self.red_male_bugs[m])
                    self.red_male_bugs[m] = new_female_bugs
                    min_score = 1
                    min_score_id = -1
                    for idx in range(self.female_bugs):
                        new_bin = self.generate_bin_solution(new_bug=new_female_bugs[:, idx])
                        # logger.info(f"Iteration S1: {s1} -- L1: {j}: red_bug [{m}, {idx}]:: {new_bin}")
                        if min_score > new_bin.score:
                            min_score = new_bin.score
                            min_score_id = idx
                            new_bin_sol = new_bin

                    self.min_fitness_ids[m] = min_score_id

                    if min_score < self.global_min_score:
                        logger.info(f"Iteration S1: {s1} -- L1: {j}: red_bug [{m}, {self.min_fitness_ids[m]}]:: MIN_G_UPDATED {min_score}")
                        self.global_min_score = min_score
                        self.glo_best_rov = new_bin_sol.rov_continous

            for j in range(self.L2):
                for m in range(self.male_bugs):
                    c3 = -0.5 + 2.7 * self.rng.random(self.bins_data.no_of_items)   
                    
                    male_x = self.red_male_bugs[m, :, self.min_fitness_ids[m]]
                    glomin = self.glo_best_rov
                    new_male_x = male_x + c3 * (glomin - male_x);
                    self.red_male_bugs[m, :, self.min_fitness_ids[m]] = new_male_x
                    new_bin = self.generate_bin_solution(new_bug=new_male_x)
                    # logger.info(f"Iteration S1: {s1} -- L2: {j}: red_bug [{m}, {self.min_fitness_ids[m]}]:: {new_bin}")
                    if new_bin.score < self.global_min_score:
                        logger.info(f"Iteration S1: {s1} -- L2: {j}: red_bug [{m}, {self.min_fitness_ids[m]}]:: MIN_G_UPDATED {new_bin.score}")
                        self.global_min_score = new_bin.score
                        self.glo_best_rov = new_bin.rov_continous

        for s2 in range(self.s2_max): 
            a = self.rng.permutation(self.male_bugs)         
            for m in range(self.male_bugs):
                c4 = 1.4 * self.rng.random(self.bins_data.no_of_items)
                d = a[m]

                male_x = self.red_male_bugs[m, :, self.min_fitness_ids[m]]
                male1_x = self.red_male_bugs[d, :, self.min_fitness_ids[d]]
                glomin = self.glo_best_rov
                new_male_x = male_x + c4 * (glomin - male1_x);
                self.red_male_bugs[m, :, self.min_fitness_ids[m]] = new_male_x
                
                new_bin = self.generate_bin_solution(new_bug=new_male_x)
                logger.info(f"Iteration S2: {s2}: red_bug [{m}, {self.min_fitness_ids[m]}]:: {new_bin}")
                if new_bin.score < self.global_min_score:
                    logger.info(f"Iteration S2: {s2}: red_bug [{m}, {self.min_fitness_ids[m]}]:: MIN_G_UPDATED {new_bin.score}")
                    self.global_min_score = new_bin.score
                    self.glo_best_rov = new_bin.rov_continous
                
    def run_fso(self, data_path):
        logger.info("Starting Firebug Swarm Optimization (FSO) Algorithm")
        self.data_path = data_path
        self.bins_data.extract_from_file(self.data_path)
        s1 = time.time()
        self.generate_init_population()
        self.main_search()
        e1 = time.time()

        best_solution = self.generate_bin_solution(final_best=True)
        logger.info(f"Time taken: {e1-s1}s")
        logger.info(f"Best Bug ===> {best_solution}")
        logger.info(f"Best Bug Bins => {best_solution.bins}")
        logger.info(f"Best Bug free capacities in bins => {best_solution.free_bin_caps}")
        
        with open("Result.txt", 'w') as result:
            result.write("<==== RESULTS ====>\n")
            result.write(f"Best minimum no of bins and Bin Efficiency by FSO (Best BUG):- {best_solution}\n")
            result.write(f"<--- Best BUG Bin Solution --->\n")
            for bin_id, bin in enumerate(best_solution.bins):
                result.write(f"Bin {bin_id + 1}: {bin}\n")
            result.write(f"Free capacities in each bins: {best_solution.free_bin_caps}\n")
       
        return best_solution, (e1 - s1)

if __name__ == "__main__":
    # path = "./../data/bin1data/N3C2W4_T.BPP"
    # path = "./../data/bin2data/N2W1B1R7.BPP"
    # path = "./../data/bin2data/N3W1B3R0.BPP"
    # path = "./../data/bin2data/N1W1B1R5.BPP"
    path = "./../data/bin3data/HARD2.BPP"
    # fso = FSO(male_bugs=20, female_bugs=5, s1_max=82,s2_max=153)
    fso = FSO(male_bugs=20, female_bugs=5, s1_max=30, s2_max=20)  #fso = FSO(male_bugs=20, female_bugs=5, s1_max=25, s2_max=40)
    fso.run_fso(path)